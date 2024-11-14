
Main_Summary = """

Please provide a thorough and detailed summary of the following meeting transcript. Structure the summary in a narrative format, ensuring that all discussions and key points are included comprehensively. Use the following sub-headings to organize the content:

Content:

Provide a concise yet detailed overview of the meeting, summarizing the most important topics discussed.
Include the purpose of the meeting, key objectives, and any significant outcomes or agreements reached during the meeting.
Ensure that important updates or announcements are highlighted, and that the overall direction of the discussions is clearly conveyed.
Content:

Provide an in-depth account of the discussions held during the meeting.
Include comments and inputs from different participants, and any questions raised or responses given.
Summarize detailed discussions around ongoing projects, challenges, issues, and clarifications provided.
Ensure that every detail is included, covering both minor points and major discussions, all in a connected and cohesive narrative.
The summary should be a single large paragraph that provides a professional and comprehensive view of the entire meeting. All key topics must be integrated seamlessly without using point-wise breakdowns.

"""



Executive_Summary = """

Please provide an executive summary of the following meeting transcript in list/point wise. 
The summary should focus only on the major and important points from the entire transcript. 

Give Response in only list of point with basic details.
Don't create any sub-heading. Response must be in concise form.

Give me each section is detailed and well-organized, using dash (-) points for clarity.
All points should be written in one or two sentences, and the format must be as follows :
"\\n - Key point or decision 1, \\n - Key point or decision 2, \\n - Key point or decision 3, \\n"

NOTE: If the transcript doesn't contain any major decisions or key points, return: "No specific decisions or future actions discussed in this transcript." Avoid adding any extra information.
"""



Meeting_Notes = """

Generate a detailed meeting summary from the following transcript. 
Include Every points, from minor to major, discussed during the meeting. 
Organize the summary under sub-headings based on topics. 
For each sub-heading, include all relevant points that relate to the same topic, even if they were discussed at different times. 
Ensure the summary is comprehensive, descriptive, and covers all topics discussed in the meeting in detail.
Don't add metadata value in summary.

Give me each section is detailed and well-organized, using dash (-) points for clarity. 
if have point have a title then wrap title with single astrict (*).
The summary should be well-structured and organized with the following format:

"\\n *Topic 1:* \\n - Information about Topic 1 \\n - Information about Topic 1 \\n - Information about Topic 1 \\n \\n *Topic 2:* - Information about Topic 2 \\n - Information about Topic 2 \\n \\m *Next Steps:* \\n - Details of next steps \\n - Details of next steps \\n

"""


Other_Key_Point = """

Please provide a summary of the other key points discussed in the following meeting transcript. 

This is Main Summary : 
Executive_Summary : {executive_summary}
Meeting_Notes : {meeting_notes}

The text above contains the Executive Summary and Meeting Notes. 
Please analyze them in comparison with the full transcript, and include any other key points from the transcript that are not covered in the Executive Summary and Meeting Notes.
Don't add Participants, Action items, etc.
Important: All Key Point in list wise, Don't create sub-heading. Only include point which is not include in Main summary.
Don't return transcript as it is.

Every key point have only one or two sentence. Don't generate long point. 
Give me each section is detailed and well-organized, using dash (-) points for clarity. if have point have a title then wrap title with single astrict (*)
The summary should be concise and follow this format:

- Additional Key Point 1, \\n
- Additional Key Point 2, \\n
- Additional Key Point 3 \\n

NOTE : Return only other key points from the summary. If the transcript doesn't cover the topic, respond with 'No specific decisions or future actions discussed in this transcript.'. 

"""


Action_Items_Template = """

Please provide a Action item of the following meeting.
Action item cover Task Description, Assignee, Due Date or Timeline, Priority Level, Follow-up, Dependencies, Resources Needed, Status or Next Steps etc.
Response must be generate in list/points. Every key point have only one or two sentence. Don't generate long point.
All the point start with dash (-).
The Action item should be well-organized and follow this format:

- Action item 1, \\n
- Action item 2, \\n
- Action item 3  \\n

NOTE : Don't given extra information in response, just return only Action Items of summary like format. \n if transcript is not able generate Action Items of above topic then only return "No specific decisions or future actions discussed in this transcript". 

"""


Tasks_and_Decisions_Prompt = """
    
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
    6) Formate of summary_responses is like "\n**Task**:\n- Task 1\n- Task 2\n- Task 3\n\n**Decision**:\n- Decision 1\n- Decision 2\n- Decision 3"
    7) Create only 2 sub-heading Task and Decision only one time.
    8) Don't add "Here are the Action Items...."

"""

