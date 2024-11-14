
Main_Summary = """

Please provide an extensive and detailed summary of the following meeting transcript. The summary should be a single, continuous paragraph that covers the entire scope of the meeting. Structure the summary with the following components:

Content Overview:

Present a detailed overview of the meeting, encapsulating the purpose, key objectives, and significant outcomes or agreements reached. Highlight any important updates or announcements, and clearly convey the overall direction of the discussions.
Detailed Discussions:

Offer a thorough account of the discussions, including ongoing projects, challenges, issues, and clarifications provided. Ensure that every aspect of the meeting is included, capturing both minor details and major points.
The summary should seamlessly integrate all key topics discussed during the meeting. Ensure that the narrative is comprehensive and includes all relevant details, providing a full picture from start to finish without breaking the summary into point-wise sections. The response should be detailed, cohesive, and professional, reflecting the depth and breadth of the meeting.
"""



Executive_Summary = """

This is Main Summary of Meeting Transcript.
MAIN SUMMARY : {main_summary}

Please provide an executive summary from the above main summary. 
The summary should focus only on the major points, decisions, and key discussions from the entire transcript. 

IMPORTANT : 
- It should be concise
- short form
- presented as a list of points
- Don't create sub-headings or categories

Give me each section is detailed and well-organized, using dash (-) points for clarity.
All the point start with dash (-).
All points should be written in one or two sentences, and the format must be as follows:

- Key point or decision 1, \\n
- Key point or decision 2, \\n
- Key point or decision 3, \\n

NOTE: If the transcript doesn't contain any major decisions or key points, return: "No specific decisions or future actions discussed in this transcript." Avoid adding any extra information.
"""



Meeting_Notes = """

This is Main Summary of Meeting Transcript.
MAIN SUMMARY : {main_summary}

Based on the provided main summary, please refine and organize the content into well-structured meeting notes. The refined meeting notes should break down the main summary into relevant sub-headings and detailed points under each section. Ensure that the content is well-organized and comprehensive.

For the meeting notes, please:

Create Sub-Headings: Identify and create appropriate sub-headings based on the key topics and themes covered in the main summary. Each sub-heading should represent a distinct topic or area of discussion from the meeting.

Detail Points Under Each Sub-Heading: Under each sub-heading, provide detailed information extracted from the main summary. Ensure that each point under the sub-heading is well-organized and provides a clear and thorough account of the related discussions, updates, or outcomes.

Ensure Comprehensive Coverage: Make sure that every aspect of the main summary is included under the relevant sub-headings. The refined notes should cover all significant details, providing a complete and detailed overview of each topic discussed in the meeting.

Give me each section is detailed and well-organized, using dash (-) points for clarity. if have point have a title then wrap title with single astrict (*).
All the point start with dash (-).
The summary should be well-structured and organized with the following format:

*Topic 1:*
- Information about Topic 1 \\n
- Information about Topic 1 \\n
- Information about Topic 1 \\n

*Topic 2:*
- Information about Topic 2 \\n
- Information about Topic 2 \\n

*Next Steps:*
- Details of next steps \\n
- Details of next steps \\n

"""


Other_Key_Point = """

Please provide a summary of the other key points discussed in the following meeting transcript. 

This is Main Summary of Meeting Transcript.
MAIN SUMMARY : {main_summary}

Based on the provided main summary and the full meeting transcript, please identify and summarize any key points from the transcript that are not covered in the main summary. The output should include all significant details and points mentioned in the transcript but omitted from the main summary.

Identify Missing Points: Review the transcript and identify all key points, discussions, updates, or details that are present in the transcript but not included in the main summary.

Important: All Key Point in list wise, Don't create sub-heading. Only include point which is not include in Main summary.

Every key point have only one or two sentence. Don't generate long point. 
Give me each section is detailed and well-organized, using dash (-) points for clarity.
All the point start with dash (-).
The summary should be concise and follow this format:

- Additional Key Point 1, \\n
- Additional Key Point 2, \\n
- Additional Key Point 3 \\n

NOTE : Return only other key points from the summary. If the transcript doesn't cover the topic, respond with 'No specific decisions or future actions discussed in this transcript.'. 

"""


action_items_template = """

Please provide a Action item of the following meeting.
Action item cover Task Description, Assignee, Due Date or Timeline, Priority Level, Follow-up, Dependencies, Resources Needed, Status or Next Steps etc.
Action Item convert into summary formate of task and decision.
Response must be generate in list/points. Every key point have only one or two sentence. Don't generate long point.
All the point start with dash (-).
The Action item should be well-organized and follow this format:

- Action item 1, \\n
- Action item 2, \\n
- Action item 3  \\n

NOTE : Don't given extra information in response, just return only Action Items of summary like format. \n if transcript is not able generate Action Items of above topic then only return "No specific decisions or future actions discussed in this transcript". 

"""


tasks_and_decisions_prompt = """
    
You have a list of Action Item from the Meeting Transcript.
Please separate Task and Decision from the above Action Item points.

If Point Include Task Description, Assignee, Due Date or Timeline, Priority Level, Resources Needed, Dependencies, Follow-up then include in Task Section.

If Point Include Final Choices or Selections, Agreements on Deadlines, Approval or Rejection, Direction or Strategy, Dispute Resolutions or Consensus, Budget and Resource Allocations then include in Decision Section.
    
Instructions:

    1) Review each Action point carefully.
    2) Extract all distinct points mentioned in the transcript.
    3) Remove any duplicate points to ensure each point is unique.
    4) Ensure the final points is professional and clearly organized.
    5) Give me each section is detailed and well-organized, using dash (-) points for clarity. 
    6) All the point start with dash (-)
    7) Formate of summary_responses is like :
        **Task**:
            - [Task 1] \\n
            - [Task 2] \\n
            - [Task 3] \\n
        
        **Decision**:
            - [Decision 1] \\n
            - [Decision 2] \\n
            - [Decision 3] \\n
            

"""

