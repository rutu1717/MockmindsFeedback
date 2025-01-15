from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

class FeedbackRequest(BaseModel):
    question: str
    user_answer: str
    reference_answer: str

class FeedbackResponse(BaseModel):
    completeness: float
    clarity: float
    technical_accuracy: float

@app.post("/evaluate-feedback", response_model=FeedbackResponse)
async def evaluate_feedback(request: FeedbackRequest):
    user_answer = request.user_answer
    reference_answer = request.reference_answer

    user_embedding = model.encode(user_answer, convert_to_tensor=True)
    reference_embedding = model.encode(reference_answer, convert_to_tensor=True)

 
    completeness_score = util.cos_sim(user_embedding, reference_embedding).item()

    clarity_score = 1.0 if len(user_answer.split()) > 5 else 0.5

    accuracy_score = util.cos_sim(user_embedding, reference_embedding).item()

    return FeedbackResponse(
        completeness=round(completeness_score, 2),
        clarity=round(clarity_score, 2),
        technical_accuracy=round(accuracy_score, 2)
    )
