Executive_Summary = """

Please provide an executive summary of the following meeting transcript. 

Content: 
- This section provides a concise overview of the meeting, summarizing the most important topics discussed. 
- It highlights the main objectives, significant outcomes, and any key agreements made during the meeting.

It Can Include Purpose of the meeting, Key points discussed, Summary of major decisions or conclusions, Important updates or announcements.

INSTRUCTION : 

Every key point have only one or two sentence. Don't generate long point. 
All the point start with dash (-).
The summary should be concise and follow this format:

- Key point or topic 1, \\n
- Key point or topic 2, \\n
- Key point or topic 3, \\n

NOTE : Don't given extra information in response, just return only executive summary key point of summary like format. 
if transcript is not able generate summary of above topic then only return "No specific decisions or future actions discussed in this transcript". 
"""

Meeting_Notes = """

Please provide detailed description of meeting notes based on the following meeting transcript. 

Content: 
- This section contains detailed notes on the discussions that took place during the meeting. 
- It includes who said what, any questions raised, and responses given. 
- This section can be more extensive, covering all the topics discussed.

What to Include:
- Detailed accounts of discussions.
- Comments and inputs from different participants.
- Explanations or clarifications provided.
- Discussions about ongoing projects, challenges, or issues.

INSTRUCTION : 

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

Content: 
- This section captures any additional points that were discussed during the meeting but do not fit neatly into the other categories. 
- It can include miscellaneous topics, observations, or items that require follow-up.

What to Include:
- Supplementary or ad-hoc discussions.
- Informal agreements or suggestions.
- Any observations or remarks made by participants.
- Additional topics that may not have been on the original agenda.

INSTRUCTION : 

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

