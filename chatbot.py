from flask import Flask, Response, request, jsonify, render_template, session , send_file
from transformers import pipeline
import math
import statistics
import re
import uuid
from gtts import gTTS
import os
# Initialize Flask app
app = Flask(__name__)
app.secret_key = str(uuid.uuid4())  # Generate a unique secret key for sessions



# Example text (Table of Contents)
text = """
Hello mr sofiane kassmi, hello everyone . i'm Your favorite Ai bot and today we will talk about statistics 
Welcome to the Interactive Statistics Bot.

Statistics is the science of collecting, analyzing, interpreting, presenting, and organizing data. It is used to make informed decisions in the presence of uncertainty. 
Statistics helps us understand and interpret patterns in data, and it plays a critical role in many fields such as business, healthcare, social sciences, and engineering.

Here are some important concepts in statistics:

1. **Descriptive Statistics**: Descriptive statistics is about summarizing or describing a set of data. This involves measures like the mean, median, mode, and standard deviation, which give insights into the data's distribution.

2. **Inferential Statistics**: Inferential statistics involves making predictions or inferences about a population based on a sample. It uses methods like hypothesis testing and confidence intervals to draw conclusions about the broader population.

3. **Theorems in Statistics**:
   - **Central Limit Theorem**: This theorem states that the distribution of sample means approaches a normal distribution as the sample size increases, regardless of the population’s distribution. This is fundamental for hypothesis testing and constructing confidence intervals.
   - **Law of Large Numbers**: The law of large numbers states that as a sample size grows, the sample mean will get closer to the population mean. This helps ensure that estimates based on large samples are accurate.

4. **Characteristics of a Good Statistical Model**:
   - **Simplicity**: A good statistical model should be simple, making it easier to understand and interpret.
   - **Accuracy**: The model should provide accurate predictions or estimates of real-world values.
   - **Robustness**: The model should perform well even with variations in the data.
   - **Relevance**: The model must consider all relevant variables and not include irrelevant ones.

5. **The Crucial Role of Statistics in AI and Data Science**:
   - In **Artificial Intelligence (AI)** and **Data Science**, statistics plays a foundational role in data analysis, model building, and decision-making. Statistical methods are used to preprocess data, identify patterns, and develop predictive models.
   - **Machine Learning**, a key component of AI, relies heavily on statistical concepts like probability, regression, and classification to develop algorithms that learn from data. Without a strong statistical foundation, models would lack the rigor necessary to draw reliable conclusions and predictions.
   - Statistical analysis is essential for **model evaluation** in AI, helping data scientists assess the performance of models and adjust parameters to improve accuracy. Techniques like cross-validation, A/B testing, and hypothesis testing are commonly used to assess and improve the robustness of AI models.
   - **Big Data** analysis is also driven by statistical methodologies, allowing data scientists to make sense of large, complex datasets and extract valuable insights that power modern AI applications, from recommendation systems to fraud detection.

Now, let's dive deeper into theorems.

- The **Central Limit Theorem** tells us that for large sample sizes, the sampling distribution of the sample mean will approximate a normal distribution, even if the original population is not normally distributed. This is a powerful result because it allows us to apply statistical techniques based on the normal distribution to many problems.
  
- The **Law of Large Numbers** explains that as the sample size increases, the sample mean becomes a more reliable estimator of the population mean. This makes larger samples more accurate in reflecting true population characteristics.

These theorems form the foundation of inferential statistics, making them essential for analyzing data and making predictions.

Thank you for using the Interactive Statistics Bot.
"""
# Specify the path where you want to save the audio file
file_path = "C:/Users/MSI/OneDrive/Desktop/math_chatbot/static/audio/table_of_contents.mp3"

# Ensure the directory exists
os.makedirs('static/audio', exist_ok=True)

# Generate the speech and save it as an MP3 file
speech = gTTS(text=text, lang='en')
speech.save(file_path)

print("Audio file generated successfully.")
# Load the BERT model for Question Answering from Hugging Face
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
@app.route('/generate_toc_speech', methods=['GET'])
def generate_toc_speech():
    try:
        # Replace with the correct file path
        with open('C:/Users/MSI/OneDrive/Desktop/math_chatbot/static/audio/table_of_contents.mp3', 'rb') as audio:
            return Response(audio.read(), mimetype='audio/mpeg')
    except FileNotFoundError:
        return "Audio file not found", 404
# Function to calculate standard deviation
def calculate_standard_deviation(data):
    return statistics.stdev(data)

# Function to calculate mean
def calculate_mean(data):
    return sum(data) / len(data)

# Function to calculate variance
def calculate_variance(data):
    return statistics.variance(data)

# Function to calculate median
def calculate_median(data):
    return statistics.median(data)

# Function to calculate correlation between two datasets
def calculate_correlation(data1, data2):
    return statistics.correlation(data1, data2)

# Initialize session storage for conversation history
def initialize_conversation_history():
    if 'conversation_history' not in session:
        session['conversation_history'] = []

# Function to add to conversation history
def add_to_conversation_history(question, answer):
    initialize_conversation_history()
    session['conversation_history'].append({
        'question': question,
        'answer': answer
    })
    # Limit conversation history to last 10 interactions
    session['conversation_history'] = session['conversation_history'][-10:]

# Function to generate detailed explanation
def generate_detailed_explanation(calculation_type, data, result):
    explanations = {
        'standard deviation': f"""
📊 Standard Deviation Analysis
------------------------------

**Definition:**
- A measure of the amount of variation or dispersion in a set of values
- Indicates how spread out numbers are from their average value

**Data Insights:**
- Data Set:        {data}
- Calculated Value: {result:.2f}

**Interpretation:**
- 🟢 Low Standard Deviation:
  * Values tend to be close to the mean
  * Suggests more consistent or clustered data

- 🔴 High Standard Deviation:
  * Values are spread out over a wider range
  * Indicates more variability and diversity in the data

**Calculation Steps:**
1️⃣ Calculate the mean of the data set
2️⃣ Calculate the squared differences from the mean
3️⃣ Take the average of those squared differences
4️⃣ Take the square root of that average

**Practical Applications:**
- Quality control in manufacturing
- Risk assessment in finance
- Performance evaluation in sports and academics
        """,
        'mean': f"""
📈 Mean (Average) Analysis
--------------------------

**Definition:**
- The sum of all values divided by the total number of values
- Represents the central tendency of a data set

**Data Insights:**
- Data Set:        {data}
- Calculated Value: {result:.2f}

**Interpretation:**
- 🎯 Central Tendency:
  * Provides a single representative value for the entire data set
  * Useful for understanding the typical or central value

- 🔍 Key Characteristics:
  * Sensitive to extreme values (outliers)
  * Can be misleading if data is skewed

**Calculation Steps:**
1️⃣ Add up all the values in the data set
2️⃣ Divide the sum by the total number of values

**Practical Applications:**
- Economic indicators
- Scientific research
- Student grade analysis
- Population demographics
        """,
        'variance': f"""
📊 Variance Analysis
-------------------

**Definition:**
- Measures how far a set of numbers is spread out from their average value
- Indicates the variability of data points around the mean

**Data Insights:**
- Data Set:        {data}
- Calculated Value: {result:.2f}

**Interpretation:**
- 🟢 Low Variance:
  * Data points are close to the mean
  * Suggests more consistent or uniform data

- 🔴 High Variance:
  * Data points are spread out
  * Indicates more diversity and dispersion in the data

**Calculation Steps:**
1️⃣ Calculate the mean of the data set
2️⃣ Calculate the squared differences from the mean
3️⃣ Take the average of those squared differences

**Practical Applications:**
- Financial risk assessment
- Investment portfolio analysis
- Quality control in manufacturing
- Performance measurement in various fields
        """,
        'median': f"""
📊 Median Analysis
-----------------

**Definition:**
- The middle value when a data set is ordered from least to greatest
- Less affected by extreme values compared to the mean

**Data Insights:**
- Data Set:        {data}
- Calculated Value: {result}

**Interpretation:**
- 🎯 Central Tendency:
  * Represents the midpoint of the data set
  * Provides a robust measure of central value

- 🔍 Key Characteristics:
  * Not influenced by outliers
  * Particularly useful for skewed distributions

**Calculation Steps:**
1️⃣ Order the data set from smallest to largest
2️⃣ If odd number of values, select the middle value
3️⃣ If even number of values, take the average of the two middle values

**Practical Applications:**
- Income and salary studies
- Real estate pricing
- Academic performance analysis
- Healthcare data interpretation
        """,
        'correlation': f"""
🔗 Correlation Analysis
----------------------

**Definition:**
- Measures the strength and direction of the relationship between two data sets
- Indicates how variables change in relation to each other

**Data Insights:**
- Data Set 1: {data[0]}
- Data Set 2: {data[1]}
- Calculated Correlation: {result:.2f}

**Correlation Interpretation:**
- 🟢 Positive Correlation (+1):
  * Perfect positive relationship
  * As one variable increases, the other increases

- 🔴 Negative Correlation (-1):
  * Perfect negative relationship
  * As one variable increases, the other decreases

- ⚪ No Correlation (0):
  * No linear relationship between variables
  * Variables change independently

**Correlation Strength:**
- 0.00 - 0.19:  Very weak
- 0.20 - 0.39:  Weak
- 0.40 - 0.59:  Moderate
- 0.60 - 0.79:  Strong
- 0.80 - 1.00:  Very strong

**Practical Applications:**
- Economic forecasting
- Scientific research
- Market trend analysis
- Social science studies
        """
    }
    return explanations.get(calculation_type, "No detailed explanation available.")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    """
    Enhanced endpoint to handle user queries with conversation memory and detailed explanations
    """
    data = request.json
    user_question = data.get("question")
    level = data.get("level", "university")

    # Check if the question involves a statistical calculation
    calculation_types = {
        'standard deviation': calculate_standard_deviation,
        'mean': calculate_mean,
        'variance': calculate_variance,
        'median': calculate_median,
        'correlation': calculate_correlation
    }

    for calc_type, calc_func in calculation_types.items():
        if calc_type in user_question.lower():
            # Find datasets in the question
            matches = re.findall(r"\[(.*?)\]", user_question)
            
            try:
                if calc_type == 'correlation':
                    if len(matches) == 2:
                        data1 = list(map(int, matches[0].split(',')))
                        data2 = list(map(int, matches[1].split(',')))
                        result = calc_func(data1, data2)
                        detailed_data = (data1, data2)
                else:
                    numbers = list(map(int, matches[0].split(',')))
                    result = calc_func(numbers)
                    detailed_data = numbers

                # Format the reply with detailed explanation
                reply = f"The {calc_type} of the data set is: {result:.2f}\n\n"
                reply += generate_detailed_explanation(calc_type, detailed_data, result)
                
                add_to_conversation_history(user_question, reply)
                return jsonify({"reply": reply, "conversation_history": session.get('conversation_history', [])})
            
            except Exception as e:
                reply = f"Sorry, I encountered an error processing your statistical calculation: {str(e)}"
                return jsonify({"reply": reply})

    # If not a statistical calculation, use QA model
    context = """
    Statistics is the branch of mathematics that deals with collecting, analyzing, interpreting, presenting, and organizing data. 
    It involves concepts such as probability, distributions, sampling, hypothesis testing, and regression analysis.
    Advanced statistical techniques include regression analysis, hypothesis testing, confidence intervals, 
    and various probability distributions like normal, binomial, and Poisson distributions.
    """
    
    try:
        # BERT QA model will try to extract the answer from the provided context
        result = qa_pipeline(question=user_question, context=context)
        answer = result['answer']
        
        # Generate a more comprehensive response
        comprehensive_reply = f"""
📚 Statistical Knowledge Exploration
------------------------------------

**Question Analyzed:** {user_question}

**Quick Answer:**
{answer}

**Detailed Explanation:**
- 🧮 Statistics is a powerful mathematical discipline
- 🔍 Helps understand and interpret complex data sets
- 📊 Provides insights across various fields of study

**Key Insights:**
- Scientific research relies heavily on statistical analysis
- Economic decisions are informed by statistical models
- Social sciences use statistics to understand human behavior

**Recommended Learning Paths:**
1️⃣ Explore basic statistical concepts
2️⃣ Learn about probability distributions
3️⃣ Study hypothesis testing techniques
4️⃣ Practice data interpretation skills

**Additional Resources:**
- Academic textbooks on statistics
- Online courses and tutorials
- Statistical software and data analysis tools
        """
        
        add_to_conversation_history(user_question, comprehensive_reply)
        return jsonify({"reply": comprehensive_reply, "conversation_history": session.get('conversation_history', [])})
    
    except Exception as e:
        reply = f"Sorry, I couldn't understand the question. Error: {str(e)}"
        return jsonify({"reply": reply})

@app.route('/conversation_history', methods=['GET'])
def get_conversation_history():
    """
    Endpoint to retrieve conversation history
    """
    initialize_conversation_history()
    return jsonify({"conversation_history": session.get('conversation_history', [])})

@app.route('/clear_history', methods=['POST'])
def clear_conversation_history():
    """
    Endpoint to clear conversation history
    """
    if 'conversation_history' in session:
        session['conversation_history'] = []
    return jsonify({"status": "Conversation history cleared"})

@app.route('/quiz', methods=['POST'])
def generate_quiz():
    """
    Enhanced quiz generation with more detailed questions and explanations
    """
    data = request.json
    level = data.get("level", "university")

    try:
        # More comprehensive and challenging quiz
        quiz = {
            "questions": [
                {
                    "question": "What is the mean of the following data set: [10, 20, 30, 40, 50]?", 
                    "options": ["25", "30", "35", "40"], 
                    "answer": "30",
                    "explanation": "Calculate the mean by adding all values and dividing by the number of values. (10 + 20 + 30 + 40 + 50) / 5 = 30"
                },
                {
                    "question": "Calculate the standard deviation of the data set: [5, 10, 15, 20, 25]", 
                    "options": ["7.07", "5", "10", "15"], 
                    "answer": "7.07",
                    "explanation": "Standard deviation measures spread around the mean. Involves calculating variance and then taking the square root."
                },
                {
                    "question": "Which statistical concept describes the probability of observing data given a null hypothesis?", 
                    "options": [
                        "Type I Error", 
                        "P-value", 
                        "Confidence Interval", 
                        "Significance Level"
                    ], 
                    "answer": "P-value",
                    "explanation": "A p-value represents the probability of obtaining test results at least as extreme as the observed results, assuming the null hypothesis is true."
                },
                {
                    "question": "What is the correct formula for sample variance?", 
                    "options": [
                        "Σ(xi - x̄)^2 / n", 
                        "Σ(xi - x̄)^2 / (n-1)", 
                        "Σxi / n", 
                        "Σxi^2 / n"
                    ], 
                    "answer": "Σ(xi - x̄)^2 / (n-1)",
                    "explanation": "Sample variance uses (n-1) in the denominator to provide an unbiased estimate of population variance."
                },
                {
                    "question": "Which probability distribution models the number of successes in fixed independent trials?", 
                    "options": [
                        "Normal distribution", 
                        "Poisson distribution", 
                        "Binomial distribution", 
                        "Exponential distribution"
                    ], 
                    "answer": "Binomial distribution",
                    "explanation": "Binomial distribution calculates probability of specific number of successes in a fixed number of independent yes/no experiments."
                }
            ]
        }
        return jsonify({"quiz": quiz})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)