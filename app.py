# app.py


import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM


# Load environment variables
load_dotenv()


# Configure LLM
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)


def code_review_optimization_squad():


    # ─────────────────────────────────────────────────────────────
    # Agent 1: Python Developer
    # ─────────────────────────────────────────────────────────────
    developer = Agent(
        role="Python Developer",
        goal="Write efficient, clean Python code for the given problem.",
        backstory=(
            "Experienced Python engineer with expertise in clean coding "
            "practices, PEP 8, and software design patterns."
        ),
        verbose=True,
        llm=llm
    )


    # ─────────────────────────────────────────────────────────────
    # Agent 2: Code Reviewer
    # ─────────────────────────────────────────────────────────────
    reviewer = Agent(
        role="Code Reviewer",
        goal="Identify improvements, refactoring opportunities, and enforce best practices.",
        backstory=(
            "Senior developer with 12+ years of experience conducting "
            "thorough code reviews for large-scale Python projects."
        ),
        verbose=True,
        llm=llm
    )


    # ─────────────────────────────────────────────────────────────
    # Agent 3: QA Specialist
    # ─────────────────────────────────────────────────────────────
    qa_agent = Agent(
        role="QA Specialist",
        goal="Validate code correctness through comprehensive test cases.",
        backstory=(
            "QA engineer specialising in automated testing, edge-case "
            "analysis, and test-driven development."
        ),
        verbose=True,
        llm=llm
    )


    # ─────────────────────────────────────────────────────────────
    # Task 1: Development
    # ─────────────────────────────────────────────────────────────
    dev_task = Task(
        description=(
            "Write a Python function to compute the factorial of a "
            "non-negative integer. Include: docstring, type hints, "
            "and handling for invalid inputs."
        ),
        expected_output=(
            "Working Python function with docstring, type hints, "
            "inline comments, and basic usage examples."
        ),
        agent=developer
    )


    # ─────────────────────────────────────────────────────────────
    # Task 2: Review
    # ─────────────────────────────────────────────────────────────
    review_task = Task(
        description=(
            "Review the provided Python factorial function. "
            "Identify any issues with style, efficiency, or best "
            "practices, and provide an optimised version."
        ),
        expected_output=(
            "A short review summary listing 3–5 observations, "
            "followed by an improved version of the code."
        ),
        agent=reviewer,
        context=[dev_task]
    )


    # ─────────────────────────────────────────────────────────────
    # Task 3: QA Testing
    # ─────────────────────────────────────────────────────────────
    qa_task = Task(
        description=(
            "Write a set of unit tests for the reviewed factorial "
            "function using unittest. Cover: normal cases, edge "
            "cases (0, 1), and invalid inputs."
        ),
        expected_output=(
            "A complete unittest.TestCase class with at least "
            "6 test methods and a brief validation summary."
        ),
        agent=qa_agent,
        context=[review_task]
    )


    # ─────────────────────────────────────────────────────────────
    # Crew
    # ─────────────────────────────────────────────────────────────
    crew = Crew(
        agents=[developer, reviewer, qa_agent],
        tasks=[dev_task, review_task, qa_task],
        process=Process.sequential,
        verbose=True
    )


    result = crew.kickoff()
    return result




if __name__ == "__main__":
    result = code_review_optimization_squad()


    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(result)
