executive_summary_template = """

Generate an executive summary of the meeting transcript . Output a concise summary in a single paragraph.

Always give response like below example.

"The meeting focused on outlining the business requirement specification for Amazon's online platform, including product listing, online shopping, customer service, and additional features. Key decisions were made to ensure the system meets user needs and business objectives, with a focus on streamlining the online shopping experience, expanding product offerings, and enhancing customer satisfaction. Critical issues raised included high traffic volumes during peak shopping periods, slow  response times, side crashes, difficulties with product search and navigation, and the need for personalized product recommendations and improved customer service. Relevant deadlines were not established, but it was agreed to use tools like Lucid Chart for process diagrams and AI to generate the B document. The meeting also discussed detailed process flows for the platform, including user registration, product listing, payment processing, and customer support."
"""

# Template for generating detailed meeting notes from a transcript
meeting_notes_template = """

Please provide detailed description of meeting notes based on the following meeting transcript.
The notes should include the main topics discussed, specific feedback given, comparisons made, and any next steps or action items identified.
Give me each section is detailed and well-organized, using dash (-) points for clarity. if have point have a title then wrap title with single astrict (*).
The notes should be well-organized and follow this format:

*Topic 1* \\n
- Details about Topic 1 \\n,
- Details about Topic 1 \\n,
- Details about Topic 1 \\n,

*Topic 2* \\n
- Details about Topic 2 \\n,
- Details about Topic 2 \\n,

*Topic 3* \\n
- Details about Topic 3 \\n,
- Details about Topic 3 \\n,

*Topic 4* \\n
- Details about Topic 4 \\n,
- Details about Topic 4 \\n,

 ...


NOTE : if transcript is not able generate summary of above topic then only return "No specific decisions or future actions discussed in this transcript".

"""

other_key_point_template = """

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

action_items_template = """

Please provide a Action item of the following meeting.
Action item cover Task, Decision, Plan, etc.
Every key point have only one or two sentence. Don't generate long point.
All the point start with dash (-).
The Action item should be well-organized and follow this format:

- Action item 1, \\n
- Action item 2, \\n
- Action item 3  \\n
- Action item 4  \\n
- so on ...

NOTE : Don't given extra information in response, just return only Action Items of summary like format. \n if transcript is not able generate Action Items of above topic then only return "No specific decisions or future actions discussed in this transcript".

"""
