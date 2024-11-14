import plantuml

def convert_to_uml(input_text):
    """
    Converts input text into PlantUML code for a sequence diagram.

    Args:
        input_text (str): Input text to convert.

    Returns:
        str: PlantUML code for the generated sequence diagram.
    """

    participants = []
    messages = []
    for line in input_text.splitlines():
        if ": " in line:
            speaker1, text = line.split(": ", 1)
            if " " in text:
                speaker2, message = text.split(" ", 1)
                participants.append(f"participant {speaker1}")
                participants.append(f"participant {speaker2}")
                messages.append(f"{speaker1} -> {speaker2}: {message}")
            else:
                participants.append(f"participant {speaker1}")
        else:
            # Ignore lines that don't start with a colon
            pass  # or handle them differently if needed

    uml_code = "@startuml\n" + "\n".join(participants) + "\n" + "\n".join(messages) + "\n@enduml"
    file='seq_1_diagram.puml'
    outfile='seq_diagram.png'
    with open(file, 'w') as f:
        f.write(uml_code)
    server = plantuml.PlantUML(url='http://www.plantuml.com/plantuml/img/')
    server.processes_file(file, outfile='seq_diagram.png')
    return outfile
