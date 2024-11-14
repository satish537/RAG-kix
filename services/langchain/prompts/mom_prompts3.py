
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

Summarize the key points and decisions from the meeting in a concise paragraph (around 100-150 words) that captures the main outcomes and action items. Please ensure the response is in a single paragraph.

Always give response like below example. 

"The meeting focused on outlining the business requirement specification for Amazon's online platform, including product listing, online shopping, customer service, and additional features. Key decisions were made to ensure the system meets user needs and business objectives, with a focus on streamlining the online shopping experience, expanding product offerings, and enhancing customer satisfaction. Critical issues raised included high traffic volumes during peak shopping periods, slow response times, side crashes, difficulties with product search and navigation, and the need for personalized product recommendations and improved customer service. Relevant deadlines were not established, but it was agreed to use tools like Lucid Chart for process diagrams and AI to generate the B document. The meeting also discussed detailed process flows for the platform, including user registration, product listing, payment processing, and customer support."

"""



Meeting_Notes = """

Generate a detailed meeting summary from the following transcript. Include every point, from minor to major, discussed during the meeting. 
Organize the summary under sub-headings based on topics. For each sub-heading, include all relevant points that relate to the same topic, even if they were discussed at different times. 
For each point and topic in discussed about timeline, include the specific date or time frame mentioned in the transcript. 
Ensure the summary is comprehensive, descriptive, and covers all topics discussed in the meeting in detail. Don't add metadata values in the summary.

Structure each section in a detailed and well-organized manner, using dash (-) points for clarity. If a point has a title, wrap the title with single asterisks (*). The summary should adhere to the following format:

\\n *Topic 1:* \\n - Information about Topic 1 \\n - Information about Topic 1 \\n - Information about Topic 1 \\n \\n *Topic 2:* \\n - Information about Topic 2 \\n - Information about Topic 2 \\n \\n *Next Steps:* \\n - Details of next steps \\n - Details of next steps \\n

"""


Other_Key_Point = """

Analyze the provided Executive Summary, Meeting Notes, and the full meeting transcript.
Identify and extract any key points from the transcript that are not included in the Executive Summary and Meeting Notes.
Present the findings in a bullet-point list format.
Ensure that only points which are absent from both the Executive Summary and Meeting Notes are included in the final output.
Use clear and concise language for each point.

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

