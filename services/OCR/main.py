import os
import re
import cv2
import json
import easyocr
import asyncio
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import threading
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

# Constants
FRAME_INTERVAL = 30
SIMILARITY_THRESHOLD = 0.88

# Global thread pool
executor = ThreadPoolExecutor(max_workers=4)

# Thread-local storage for EasyOCR readers
thread_local = threading.local()

def get_ocr_reader():
    """Get thread-local EasyOCR reader instance"""
    if not hasattr(thread_local, 'reader'):
        thread_local.reader = easyocr.Reader(['en'])
    return thread_local.reader

async def extract_text_by_video(fullpath):
    """
    Simple async function to extract conversation from video file
    
    Args:
        fullpath (str): Full path to the video file
        
    Returns:
        str: Extracted conversation as string with newlines
    """
    if not os.path.exists(fullpath):
        raise FileNotFoundError(f"Video file not found: {fullpath}")
    
    if not fullpath.lower().endswith('.mp4'):
        raise ValueError("Only MP4 files are allowed")
    
    # Run OCR extraction in thread pool
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(executor, extract_text_by_frame, fullpath)
    
    # Detect platform and extract conversation in thread pool
    conversation = await loop.run_in_executor(executor, extract_unique_chat_content, results, fullpath)
    
    return conversation

def extract_text_by_frame(video_path, frame_interval=FRAME_INTERVAL):
    """Extract text from video frames using OCR"""
    cap = cv2.VideoCapture(video_path)
    reader = get_ocr_reader()
    frame_count = 0
    results = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ocr_result = reader.readtext(frame_rgb)
            texts = [d[1] for d in ocr_result if d[2] > 0.5]
            results[str(frame_count)] = texts
        frame_count += 1

    cap.release()
    
    return results

def detect_platform(results):
    """
    Detect whether the video contains ChatGPT, Claude, or Meta AI conversation
    
    Args:
        results (dict): OCR results from frames
        
    Returns:
        str: 'chatgpt', 'claude', 'meta_ai', or 'unknown'
    """
    chatgpt_indicators = []
    claude_indicators = []
    meta_ai_indicators = []
    
    for frame_data in results.values():
        for line in frame_data:
            line_lower = line.lower()
            
            # ChatGPT indicators
            if "chatgpt" in line_lower:
                chatgpt_indicators.append(line)
            elif "ask anything" in line_lower:
                chatgpt_indicators.append(line)
            
            # Claude indicators
            elif line_lower.startswith("claude"):
                claude_indicators.append(line)
            elif "reply to claude" in line_lower:
                claude_indicators.append(line)
            
            # Meta AI indicators
            elif "reply to meta ai" in line_lower:
                meta_ai_indicators.append(line)
            elif "meta ai" in line_lower:
                meta_ai_indicators.append(line)
            elif "ask meta ai" in line_lower:
                meta_ai_indicators.append(line)
    
    # Decide based on frequency
    if len(meta_ai_indicators) > len(claude_indicators) and len(meta_ai_indicators) > len(chatgpt_indicators):
        return 'meta_ai'
    elif len(claude_indicators) > len(chatgpt_indicators):
        return 'claude'
    elif len(chatgpt_indicators) > 0:
        return 'chatgpt'
    else:
        return 'unknown'

def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def clean_ocr_text(text):
    """Clean OCR text"""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"\/\[\]]', '', text)
    return text

def is_better_version(new_line, existing_line):
    """Determine if new_line is a better version than existing_line"""
    if len(new_line) > len(existing_line) * 1.2:
        return True
    
    new_special = len(re.findall(r'[^\w\s.,!?;:()]', new_line))
    existing_special = len(re.findall(r'[^\w\s.,!?;:()]', existing_line))
    
    if new_special < existing_special:
        return True
    
    if new_line and existing_line and new_line[0].isupper() and not existing_line[0].isupper():
        return True
    
    return False

def extract_chatgpt_conversation_blocks(results):
    """Extract ChatGPT conversation blocks from OCR results"""
    all_content = []
    
    for frame in sorted(results, key=lambda x: int(x)):
        lines = results[frame]
        if not lines:
            continue
            
        chatgpt_idx = -1
        for i, line in enumerate(lines[:10]):
            if "chatgpt" in line.lower():
                chatgpt_idx = i
                break
                
        ask_anything_idx = -1
        for i, line in enumerate(lines[-10:]):
            if "ask anything" in line.lower():
                ask_anything_idx = len(lines) - 10 + i
                break
                
        if chatgpt_idx != -1 and ask_anything_idx != -1:
            frame_content = []
            for line in lines[chatgpt_idx + 1:ask_anything_idx]:
                cleaned = clean_ocr_text(line)
                if cleaned and len(cleaned) > 2:
                    if cleaned.lower() not in ['search', 'sign up', 'sending', 'more']:
                        frame_content.append(cleaned)
            
            if frame_content:
                all_content.append({
                    'frame': int(frame),
                    'content': frame_content
                })
    
    return all_content

def extract_claude_conversation_blocks(results):
    """Extract Claude conversation blocks from OCR results"""
    all_content = []
    
    for frame in sorted(results, key=lambda x: int(x)):
        lines = results[frame]
        if not lines:
            continue
            
        claude_start_idx = -1
        # Look for lines starting with "Claude" (case insensitive)
        for i, line in enumerate(lines[:10]):
            if line.lower().startswith("claude"):
                claude_start_idx = i
                break
                
        reply_to_claude_idx = -1
        # Look for lines containing "Reply to Claude" with optional ending dot
        for i, line in enumerate(lines[-10:]):
            line_lower = line.lower()
            if "reply to claude" in line_lower:
                reply_to_claude_idx = len(lines) - 10 + i
                break
                
        if claude_start_idx != -1 and reply_to_claude_idx != -1:
            frame_content = []
            for line in lines[claude_start_idx + 1:reply_to_claude_idx]:
                cleaned = clean_ocr_text(line)
                if cleaned and len(cleaned) > 2:
                    # Skip common UI elements
                    if cleaned.lower() not in ['search', 'sign up', 'sending', 'more', 'new chat', 'copy', 'share']:
                        frame_content.append(cleaned)
            
            if frame_content:
                all_content.append({
                    'frame': int(frame),
                    'content': frame_content
                })
    
    return all_content

def extract_meta_ai_conversation_blocks(results):
    """Extract Meta AI conversation blocks from OCR results"""
    all_content = []
    
    for frame in sorted(results, key=lambda x: int(x)):
        lines = results[frame]
        if not lines:
            continue
        
        # Check if this frame contains Meta AI indicators
        has_meta_indicators = False
        for line in lines:
            line_lower = line.lower()
            if "reply to meta ai" in line_lower or "meta ai" in line_lower:
                has_meta_indicators = True
                break
        
        if not has_meta_indicators:
            continue
        
        # Meta AI has no consistent starting word, so start from beginning
        start_idx = 0
        reply_to_meta_idx = -1
        
        # Look for "Reply to Meta AI" in last 10 lines
        for i, line in enumerate(lines[-10:]):
            line_lower = line.lower()
            if "reply to meta ai" in line_lower:
                reply_to_meta_idx = len(lines) - 10 + i
                break
        
        if reply_to_meta_idx != -1:
            frame_content = []
            
            # Extract content between markers
            for line in lines[start_idx:reply_to_meta_idx]:
                cleaned = clean_ocr_text(line)
                
                if cleaned and len(cleaned) > 2:
                    # Skip Meta AI specific UI elements
                    if not is_meta_ui_element(cleaned):
                        frame_content.append(cleaned)
            
            if frame_content:
                all_content.append({
                    'frame': int(frame),
                    'content': frame_content
                })
    
    return all_content

def is_meta_ui_element(text):
    """Check if text is a Meta AI UI element that should be skipped"""
    text_lower = text.lower()
    
    # Skip Meta AI specific UI elements
    meta_ui_elements = [
        'home', 'discover', 'create', 'notifications', 'messages',
        'imagine', 'ask meta ai anything', 'try asking about',
        'what would you like to create', 'recent conversations',
        'regenerate', 'stop generating', 'meta',
        'meta ai can make mistakes', 'try again', 'facebook',
        'search', 'sign up', 'sending', 'more'
    ]
    
    for element in meta_ui_elements:
        if element in text_lower:
            return True
    
    return False

def build_sequential_conversation(content_blocks):
    """Build sequential conversation with smart deduplication"""
    if not content_blocks:
        return []

    content_blocks.sort(key=lambda x: x['frame'])
    conversation_lines = []
    
    for block in content_blocks:
        for line in block['content']:
            found_similar = False
            
            for i, existing_line in enumerate(conversation_lines):
                sim_score = similarity(line, existing_line)
                
                if sim_score > SIMILARITY_THRESHOLD:
                    if is_better_version(line, existing_line):
                        conversation_lines[i] = line
                    found_similar = True
                    break
            
            if not found_similar:
                conversation_lines.append(line)
    
    return conversation_lines

def should_merge_lines(line1, line2):
    """Determine if two lines should be merged"""
    if re.match(r'^\d+\.', line2) or line2.startswith(('•', '-', '*')):
        return False
    
    if abs(len(line1) - len(line2)) > max(len(line1), len(line2)) * 0.7:
        return False
    
    if not line1.rstrip().endswith(('.', '!', '?', ':')) and line2 and not line2[0].isupper():
        return True
    
    return False

def post_process_conversation(lines):
    """Post-process conversation for better readability"""
    if not lines:
        return []
    
    processed_lines = []
    i = 0
    
    while i < len(lines):
        current_line = lines[i].strip()
        
        if len(current_line) < 3:
            i += 1
            continue
        
        if i + 1 < len(lines) and should_merge_lines(current_line, lines[i + 1]):
            merged_line = current_line + " " + lines[i + 1].strip()
            processed_lines.append(merged_line)
            i += 2
        else:
            processed_lines.append(current_line)
            i += 1
    
    return processed_lines

def extract_unique_chat_content(results, video_path):
    """
    Extract conversation content and return as string
    Automatically detects whether it's ChatGPT, Claude, or Meta AI
    
    Args:
        results (dict): OCR results from frames
        video_path (str): Original video path for unique naming
        
    Returns:
        str: Extracted conversation as string with newlines
    """
    # Detect platform
    platform = detect_platform(results)
    
    if platform == 'chatgpt':
        print(f"🤖 Detected ChatGPT conversation in {os.path.basename(video_path)}")
        content_blocks = extract_chatgpt_conversation_blocks(results)
    elif platform == 'claude':
        print(f"🔮 Detected Claude conversation in {os.path.basename(video_path)}")
        content_blocks = extract_claude_conversation_blocks(results)
    elif platform == 'meta_ai':
        print(f"🦾 Detected Meta AI conversation in {os.path.basename(video_path)}")
        content_blocks = extract_meta_ai_conversation_blocks(results)
    else:
        print(f"❓ Could not detect platform type in {os.path.basename(video_path)}, trying all methods")
        # Try all methods and use the one that returns more content
        chatgpt_blocks = extract_chatgpt_conversation_blocks(results)
        claude_blocks = extract_claude_conversation_blocks(results)
        meta_ai_blocks = extract_meta_ai_conversation_blocks(results)
        
        # Choose the method with most content
        if len(meta_ai_blocks) > len(claude_blocks) and len(meta_ai_blocks) > len(chatgpt_blocks):
            content_blocks = meta_ai_blocks
            platform = 'meta_ai'
            print("📝 Using Meta AI extraction method (more content found)")
        elif len(claude_blocks) > len(chatgpt_blocks):
            content_blocks = claude_blocks
            platform = 'claude'
            print("📝 Using Claude extraction method (more content found)")
        else:
            content_blocks = chatgpt_blocks
            platform = 'chatgpt'
            print("📝 Using ChatGPT extraction method (more content found)")
    
    if not content_blocks:
        print("⚠️  No conversation content found!")
        return ""
    
    conversation_lines = build_sequential_conversation(content_blocks)
    final_lines = post_process_conversation(conversation_lines)
    
    if final_lines:
        final_text = "\n".join(final_lines)
        return final_text
    else:
        print("⚠️  No final conversation lines after processing!")
        return ""

