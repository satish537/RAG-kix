
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

Generate an executive summary that captures the major points of the provided content.
Each point should be conveyed in 1-2 concise sentences, ensuring clarity and brevity.
The summary should provide a comprehensive overview that allows the reader to grasp the essential information quickly without delving into detailed explanations.
Focus on the key themes, findings, and recommendations, presenting them in a cohesive paragraph format.

Always give response like below example. 

"The team reviewed the quarterly sales performance, noting a 15% increase in revenue compared to the previous quarter, driven mainly by the launch of the new product line. Key decisions included allocating additional resources for marketing efforts and setting a target to enhance customer engagement through social media channels. Critical issues raised included supply chain disruptions affecting inventory levels, with a deadline established for the operations team to provide a mitigation plan by the end of the month. The discussion concluded with a commitment to follow up on the sales strategy adjustments in the next meeting."

"""



Meeting_Notes = """

You are tasked with generating a detailed meeting summary from the provided transcript.

Follow these guidelines to ensure a comprehensive and organized output: 

1.**Sub-Headings Organization:** - Create distinct sub-headings for each major topic discussed in the meeting.- Ensure that the sub-headings are clear and accurately reflect the content of the discussions.

2.**Detailed Point Inclusion:** - Under each sub-heading, list all relevant points discussed, regardless of their significance (minor or major).- Each point should be described in detail, capturing the essence of the discussion.

3.**Timeline Specification:** - For every point mentioned, identify and include the specific date or time frame referenced in the transcript.- Ensure that this information is clearly indicated alongside the corresponding point.

4.**Comprehensive Coverage:** - Review the entire transcript thoroughly to ensure that no topic or point is overlooked.- Aim for a summary that is both comprehensive and descriptive, providing depth to each topic discussed.

5.**Formatting:** - Structure each section in a detailed and well-organized manner, using dash (-) points for clarity. If a point has a title, wrap the title with single asterisks (*). The summary should adhere to the following format:
\\n *Topic 1:* \\n - Information about Topic 1 \\n - Information about Topic 1 \\n - Information about Topic 1 \\n \\n *Topic 2:* \\n - Information about Topic 2 \\n - Information about Topic 2 \\n \\n *Next Steps:* \\n - Details of next steps \\n - Details of next steps \\n

"""


Other_Key_Point = """

Extract key points from the provided meeting transcript that are not covered in the executive summary and meeting notes.
The response should be a concise list of unique insights, observations of the meeting.
Ensure that the summary focuses solely on information absent from the executive summary and meeting notes.
Limit the summary to a small size for clarity and brevity.
Present the findings in a point/list format without additional sub-headings or explanations.

This is Main Summary : 
EXECUTIVE_SUMMARY : {executive_summary}
MEETING_NOTES : {meeting_notes}

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

