import numpy as np



##### NOTE: Using Individualized Bayesian Knowledge Tracing Models (doi: 10.1007/978-3-642-39112-5_18) Section and Line Numbers to source parameter names and conventions #####


class BayesianKnowlegeTracing():
    def __init__(self, p_L_0: np.float64, p_T: np.float64, p_S:np.float64, p_guess: np.float64):
        """
        Parameters:
        
        P(L_0): probability of student knowing the skill before AKA p-init(2.1 l2)
        P(T): probability of a student's knowledge of a skill transitioning from known to not known after being given an opportunity to apply it AKA P(transit) (2.1 l2-3)
        P(S): probability to make a mistake when appying a known skill AKA p-slip(2.1 l4)
        P(G): probability of correctly applying a not-known skill aka p_guess (2.1 l6)

        """

        self.initial_prob = p_init
        self.transit_prob = p_transit
        self.slip_prob = p_slip
        self.guess_prob = p_guess

        #init of student's knowledge state
        self.knowledge_prob = self.initial_prob

        def update(self, correct):
            """
            Parameters:
            
            correct: Boolean (0 or 1) value that is true if the student already knew the answer (answered it correctly) and false if the student did not know the answer (answered it incorrectly)

            Returns: 

            self.knowledge_prob: float value that is the updated probability of the student knowing the skill. 
            """
            # calculate probability that student knew the skill before the observation
            if correct:
                likelihood = self.knowledge_prob * (1- self.slip_prob) + (1 - self.knowledge_prob) * self.guess_prob
            else:
                likelihood = self.knowledge_prob * self.slip_prob + (1 - self.knowledge_prob) * (1 - self.guess_prob)
            posterior = self.knowledge_prob if correct else (1- self.knowledge_prob)
            # update prob that student learns the skill
            self.knowledge_prob = posterior * likelihood / self.knowledge_prob
            # update prob of the possibility to "learn" after the observation. 
            self.knowledge_prob = self.knowledge_prob + (1 - self.knowledge_prob) * self.transit_prob
            return self.knowledge_prob

        def get_knowledge_prob(self):
            """
            Parameters: None
            Returns: Probability of a student knowing the skill
            """
            return self.knowledge_prob

