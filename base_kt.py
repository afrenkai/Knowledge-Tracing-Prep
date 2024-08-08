import numpy as np
from sklearn.linear_model import LogisticRegression

class KnowledgeTracing:
    def __init__(self, num_skills, num_questions):
        """
        Initialization for the KnowledgeTracing obj.

        Arguments:
            num_skills: number of skills
            num_questions: number of questions
        """
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.student_knowledge = np.zeros((1, num_skills))
        self.question_skills = np.zeros((num_questions, num_skills))
        self.question_difficulty = np.zeros(num_questions)

    def add_question(self, question_id, skills, difficulty):
        """
        add a question to the knowledge tracing obj.
        Arguments:
             question_id: id of the question to be used in a dict/list
             skills: list/np.ndarray of skills
             difficulty: list/np.ndarray of difficulties
        """
        self.question_skills[question_id] = skills
        self.question_difficulty[question_id] = difficulty

    def update_student_knowledge(self, responses):
        """
         Update the student's knowledge based on their responses to the questions.
        Arguments:
            responses (list or numpy.ndarray): A list or numpy array of the student's responses (1 for correct, 0 for incorrect).
        """
        X = self.question_skills
        y = responses
        model = LogisticRegression()
        model.fit(X, y)
        self.student_knowledge = model.coef_

    def predict_response(self, question_id):
        """
        Predicts the student knowledge based on their responses to the questions.
        Arguments:
             question_id: id of the question to be used in the list passed from the init
        Returns:
            probability: probability of being correct using the logreg formula (L/(1 + e^(-k(x-x_0))
            where L is the max of the curve, k is the growth rate, x_0 is the midpoint, and x is the real number
        """
        question_skill = self.question_skills[question_id]
        probability = 1 / (1 + np.exp(-(self.student_knowledge @ question_skill - self.question_difficulty[question_id])))
        return probability
