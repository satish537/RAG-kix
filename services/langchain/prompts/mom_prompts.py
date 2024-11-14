Executive_Summary = """

Please provide an executive summary of the following meeting transcript. 
The summary should include the key points discussed, decisions made, and any future actions or plans mentioned. 
Every key point have only one or two sentence. Don't generate long point. 
All the point start with dash (-).
The summary should be concise and follow this format:

- Key point or topic 1, \\n
- Key point or topic 2, \\n
- Key point or topic 3, \\n

NOTE : Don't given extra information in response, just return only executive summary key point of summary like format. \n if transcript is not able generate summary of above topic then only return "No specific decisions or future actions discussed in this transcript". 

"""

Meeting_Notes = """

Please provide detailed description of meeting notes based on the following meeting transcript. 
The notes should include the any topics discussed, specific feedback given, comparisons made, and any next steps or action items identified. 
Give me each section is detailed and well-organized, using dash (-) points for clarity. if have point have a title then wrap title with single astrict (*).
The notes should be well-organized and follow this format:

*Topic 1* \\n 
- Details about Topic 1 \\n, 
- Details about Topic 1 \\n, 
- Details about Topic 1 \\n,

*Topic 2* \\n 
- Details about Topic 2 \\n, 
- Details about Topic 2 \\n,

*Next Steps* \\n, 
- Details about Next Steps \\n, 
- Details about Next Steps \\n,
 ...


NOTE : if transcript is not able generate summary of above topic then only return "No specific decisions or future actions discussed in this transcript".

"""


Other_Key_Point = """

Please provide a summary of the other key points discussed in the following meeting transcript. 
The summary should include additional important points that were not covered in the main discussion, focusing on suggestions, priorities, future plans, and enhancements. 
Every key point have only one or two sentence. Don't generate long point. 
All the point start with dash (-).
The summary should be concise and follow this format:

- Additional Key Point 1, \\n
- Additional Key Point 2, \\n
- Additional Key Point 3 \\n

NOTE : Don't given extra information in response, just return only other key point of summary like format. \n if transcript is not able generate summary of above topic then only return "No specific decisions or future actions discussed in this transcript". 

"""


Action_Items_Template = """

Please provide a Action item of the following meeting.
Action item cover Task, Decision, Plan, etc.
Every key point have only one or two sentence. Don't generate long point.
All the point start with dash (-).
The Action item should be well-organized and follow this format:

- Action item 1, \\n
- Action item 2, \\n
- Action item 3  \\n

NOTE : Don't given extra information in response, just return only Action Items of summary like format. \n if transcript is not able generate Action Items of above topic then only return "No specific decisions or future actions discussed in this transcript". 

"""


Tasks_and_Decisions_Prompt = """
    
    You have a list of Action Item from the Meeting Transcript.
    Please separate Task and Decision from the above Action Item points.one Action Point going to only one section(Task, Decision).
     
    Instructions:

        1) Review each Action point carefully.
        2) Extract all distinct points mentioned in the transcript.
        3) Remove any duplicate points to ensure each point is unique.
        4) Ensure the final points is professional and clearly organized.
        5) Give me each section is detailed and well-organized, using dash (-) points for clarity. 
        6) Formate of summary_responses is like :
            **Task**:
                - [Task 1]
                - [Task 2]
                - [Task 3]
            
            **Decision**:
                - [Decision 1]
                - [Decision 2]
                - [Decision 3]
                
    
    """

