from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import anthropic
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    task = request.json["task"]
    level = request.json["level"]
    edu_safe = request.json.get("edu_safe", False)
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate():
        with client.messages.stream(
            model="claude-opus-4-5",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""You are nAIgate, an AI tool recommendation engine. A user needs help with the following task:

TASK: {task}
USER LEVEL: {level}
EDUCATION SAFE MODE: {edu_safe}

If EDUCATION SAFE MODE is True, only recommend tools that are appropriate for students and classrooms, have data privacy protections, and are free or low cost. Flag education_safe as true only for tools that meet these criteria.

Respond ONLY in this exact JSON format...
TASK: {task}
USER LEVEL: {level}

Respond ONLY in this exact JSON format with no extra text:
{{
  "best_match": {{
    "name": "Tool name",
    "logo_emoji": "relevant emoji",
    "cost": "Free / $X/mo / Free tier",
    "type": "Executes" or "Advises",
    "education_safe": true or false,
    "reason": "One sentence why this is the best tool for this task",
    "prompt": "A ready-to-use prompt the user can copy and paste directly into this tool to complete their task"
  }},
  "alternatives": [
    {{
      "name": "Tool name",
      "best_for": "One short phrase",
      "cost": "Free / $X/mo",
      "type": "Executes" or "Advises"
    }},
    {{
      "name": "Tool name", 
      "best_for": "One short phrase",
      "cost": "Free / $X/mo",
      "type": "Executes" or "Advises"
    }},
    {{
      "name": "Tool name",
      "best_for": "One short phrase", 
      "cost": "Free / $X/mo",
      "type": "Executes" or "Advises"
    }}
  ]
}}

Important rules:
- Only recommend REAL tools that actually exist
- "Executes" means the tool actually completes the task, "Advises" means it gives instructions
- If user level is Beginner, prioritize free and easy to use tools
- Make the prompt specific and ready to use, not generic"""
            }]
        ) as stream:
            for text in stream.text_stream:
                yield text

    return Response(stream_with_context(generate()), mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True)