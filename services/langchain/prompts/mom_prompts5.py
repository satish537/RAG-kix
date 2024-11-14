Main_Summary = """

Please generate a comprehensive and cohesive summary of the meeting transcript. The summary should include the following sections in paragraph form:

- **Purpose of the Meeting**: A brief overview of the meeting's purpose and goals.
- **Key Discussions and Updates**: A detailed account of the major discussions, updates, and announcements.
- **Outcomes and Decisions**: Outline any agreements, conclusions, or resolutions reached.
- **Next Steps or Action Items**: Mention any specific next steps or tasks identified for follow-up.

Ensure that the summary is professional, concise, and captures the full scope of the meeting, integrating both key points and minor details where necessary.

"""


Executive_Summary = """
This is a summary of the key points and decisions made during the meeting. The output should be presented as a concise list of bullet points, highlighting only major takeaways and important discussions.

IMPORTANT:
- Keep each point short and direct, no more than 1-2 sentences.
- Structure it using bullet points, without creating sub-headings or categories.

Format:
- Key point or decision 1, \n
- Key point or decision 2, \n
- Key point or decision 3, \n

If no major decisions or points are found, return: 'No specific decisions or future actions discussed in this transcript.'

"""


Meeting_Notes = """

Please generate well-structured meeting notes based on the transcript. These notes should be divided into sections with sub-headings, clearly organizing the meeting discussions.

For each section:
- **Sub-Headings**: Create sub-headings based on distinct topics or areas of discussion.
- **Details Under Each Sub-Heading**: Provide 1-2 detailed points that summarize key discussions, updates, or decisions.

Ensure that every part of the meeting is covered in the notes, with the content organized and easy to follow. The format should be as follows:

*Topic 1:*
- Detailed discussion point 1 \n
- Detailed discussion point 2 \n

*Topic 2:*
- Detailed discussion point 1 \n
- Detailed discussion point 2 \n

"""

Other_Key_Point = """

Please identify any additional key points or discussions that were not covered in the main summary of the meeting transcript. Each key point should be concise and should highlight important discussions not already included.

Each point should be 1-2 sentences, written as a bullet point.

Format:
- Additional Key Point 1, \n
- Additional Key Point 2, \n
- Additional Key Point 3, \n

If there are no additional key points to be found, return: 'No other key points discussed in this transcript.'
"""


Action_Items_Template = """

Please list the action items discussed in the meeting transcript. For each action item, provide the task description, assignee, timeline or deadline, and any priority or dependencies.

IMPORTANT:
- Present each action item as a bullet point.
- Each action item should be concise, 1-2 sentences.
- Do not include additional or unnecessary information.

Format:
- Action Item 1: Task description, Assignee, Timeline. \n
- Action Item 2: Task description, Assignee, Timeline. \n
- Action Item 3: Task description, Assignee, Timeline. \n

If no action items were discussed, return: 'No specific actions or tasks discussed in this transcript.'

"""



Tasks_and_Decisions_Prompt = """

From the provided action items, separate the tasks from the decisions. Include each task or decision under the appropriate category.

- **Tasks**: Tasks that include a description, assignee, deadline, priority, or any other actionable detail.
- **Decisions**: Decisions that reflect any final choices, agreements, approvals, or strategies decided upon during the meeting.

Format:
**Tasks**:
- Task 1: Task description \n
- Task 2: Task description \n

**Decisions**:
- Decision 1: Decision description \n
- Decision 2: Decision description \n

"""