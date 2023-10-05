import pandas as pd
import plotly.express as px
from fastapi import FastAPI
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer, util
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import json
from fastapi_cors import CORS


model = SentenceTransformer('model')

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORS,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}


@app.get("/similarity")
async def similarity(text1: str, text2: str):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)

    # Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    
    return {"similarity": cosine_scores.item()}

@app.get("/student/report")
async def student_report(data):
    # convert data from string to list
    data = json.loads(data)
    exams = data['exams']
    name = data['name']
    email = data['email']
    number = data['phone_number']
    school = data['school']
    student_id = data['id']
    

    df = pd.DataFrame(exams)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    df['taken_at'] = pd.to_datetime(df['taken_at'])

    df.fillna(0, inplace=True)
    score_percentage = px.bar(df, x='title', y='score_percentage', title='Score Percentage')
    score_percentage.update_layout(
        xaxis_title="Exam Title",
        yaxis_title="Score Percentage",
        font=dict(
            size=14,
        )
    )
    score_percentage.update_traces(texttemplate='%{y:.2f}%', textposition='inside')

    # Calculate the overall percentage of correct MCQ and open-ended questions
    if 'answered_mcq_count' in df.columns:
        if(df['answered_mcq_count'].sum() > 0):
            average_percentage_correct_mcq = (((df['answered_mcq_count'] - df['wrong_mcq_count']).sum() / df['answered_mcq_count'].sum()) * 100)
    else:
        average_percentage_correct_mcq = 0
    
    if 'answered_open_ended_count' in df.columns:
        if(df['answered_open_ended_count'].sum() > 0):
            average_percentage_correct_open_ended = (((df['answered_open_ended_count'] - df['wrong_open_ended_count']).sum() / df['answered_open_ended_count'].sum()) * 100)
    else:
        average_percentage_correct_open_ended = 0

    data = {
        'Question Type': ['MCQ', 'Open-Ended'],
        'Average Percentage Correct': [average_percentage_correct_mcq, average_percentage_correct_open_ended],
        'Percentage Text': [f'{average_percentage_correct_mcq:.2f}%', f'{average_percentage_correct_open_ended:.2f}%']
    }

    print(average_percentage_correct_mcq, average_percentage_correct_open_ended)

    avg_type_performance = px.bar(data, x='Question Type', y='Average Percentage Correct', text='Percentage Text', title='Average Performance in MCQ vs. Open-Ended Questions')

    # Customize text position
    avg_type_performance.update_traces()
    avg_type_performance.update_layout(
        font=dict(
            size=14,
        ),
        bargap=0.7
    )

    # Calculate the average score percentage across all exams
    average_score_percentage = df['score_percentage'].mean()

    # Create a bar plot for the average performance with percentage text
    avg_performance = px.bar(x=['Average Performance'], y=[average_score_percentage], title='Average Performance (Average Score Percentage)')
    avg_performance.update_layout(
        xaxis_title="",
        yaxis_title="Average Score Percentage",
        font=dict(
            size=14,
        ),
        bargap=0.8
    )
    avg_performance.update_traces(texttemplate='%{y:.2f}%', textposition='inside')

    figs = [score_percentage, avg_type_performance, avg_performance]
    images = []
    for fig in figs:
        img = BytesIO()
        fig.write_image(img)
        images.append(Image(img, width=400, height=300))


    doc = SimpleDocTemplate(f'{student_id}-report.pdf')
    styles = getSampleStyleSheet()
    title = Paragraph("Student Report", styles['h1'])
    profile = Paragraph(f"Name: {name} <br/> Number: {number} <br/> Email: {email} <br/> School: {school}", styles['Normal'])
    

    report = [title, profile]
    for image in images:
        report.append(image)
    doc.build(report)

    return FileResponse(f'{student_id}-report.pdf')

