import random
from typing import List


class GeneticEvolver:
    def __init__(
        self,
        initial_population: List[str],
        generations: int = 10,
        mutation_rate: float = 0.1,
    ):
        self.initial_population = initial_population
        self.generations = generations
        self.mutation_rate = mutation_rate

    def evolve(self):
        population = self.initial_population
        for generation in range(self.generations):
            fitness_scores = [self.evaluate_prompt(prompt) for prompt in population]
            selected_parents = self.select_parents(population, fitness_scores)
            new_population = []
            while len(new_population) < len(population):
                parent1, parent2 = random.sample(selected_parents, 2)
                child = self.crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self.mutate_prompt(child)
                new_population.append(child)

    def crossover(self, parent1, parent2):
        # Simple crossover: concatenate the first half of parent1 with the second half of parent2
        return parent1[: len(parent1) // 2] + parent2[len(parent2) // 2 :]

    def mutate_prompt(self, prompt):
        # Example mutation: add a random word
        words = ["optimize", "enhance", "improve", "boost"]
        return prompt + " " + random.choice(words)

    # Define a function to evaluate the prompt using an LLM-based evaluation function
    def evaluate_prompt(self, prompt):
        # Placeholder for LLM-based evaluation logic
        # Return a score representing the prompt's success
        return random.uniform(0, 1)  # Example: random score for demonstration

    def select_parents(self, population, fitness_scores):
        # Select top 50% based on fitness
        sorted_population = [
            x for _, x in sorted(zip(fitness_scores, population), reverse=True)
        ]
        return sorted_population[: len(population) // 2]
