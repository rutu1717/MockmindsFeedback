from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing import List

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the input schema for a single feedback item
class FeedbackRequest(BaseModel):
    question: str
    user_answer: str
    reference_answer: str

# Define the output schema for the feedback response
class FeedbackResponse(BaseModel):
    completeness: float
    clarity: float
    technical_accuracy: float

# Define the endpoint to handle batch feedback
@app.post("/evaluate-feedback-batch", response_model=FeedbackResponse)
async def evaluate_feedback_batch(feedback: List[FeedbackRequest]):
    completeness_total = 0
    clarity_total = 0
    accuracy_total = 0
    num_feedback = len(feedback)

    for request in feedback:
        user_answer = request.user_answer
        reference_answer = request.reference_answer

        # Generate embeddings for user and reference answers
        user_embedding = model.encode(user_answer, convert_to_tensor=True)
        reference_embedding = model.encode(reference_answer, convert_to_tensor=True)

        # Completeness: Calculate cosine similarity
        completeness_score = util.cos_sim(user_embedding, reference_embedding).item()

        # Clarity: Simplified check
        clarity_score = 1.0 if len(user_answer.split()) > 5 else 0.5

        # Technical Accuracy: Compare embeddings
        accuracy_score = util.cos_sim(user_embedding, reference_embedding).item()

        # Accumulate scores for averaging
        completeness_total += completeness_score
        clarity_total += clarity_score
        accuracy_total += accuracy_score

    # Calculate averages
    return FeedbackResponse(
        completeness=round(completeness_total / num_feedback, 2),
        clarity=round(clarity_total / num_feedback, 2),
        technical_accuracy=round(accuracy_total / num_feedback, 2)
    )
