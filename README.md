# Business Analytics System

This project showcases the power of Azure AI integrated with GPT-4 to deliver an advanced AI-driven experience. Leveraging Azure’s powerful infrastructure and the capabilities of GPT-4 for text processing, the application provides [Python-based Business Analytics System that leverages financial data, predictive analytics, and reporting tools to generate insights, visualizations, and PDF reports for business analysis.].

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Sample Results](#sample-results)

---

## Project Overview
The Business Analytics System is designed to gather sales and market data, make revenue predictions, and generate visualizations and PDF reports. This system also supports email delivery of reports for easy distribution.

## Features
### Azure AI Integration
This project takes advantage of Azure’s AI suite, ensuring scalable, secure, and high-performance AI functionalities. Through Azure, the system supports efficient data handling, real-time processing, and advanced analytics, setting the groundwork for sophisticated AI workflows.

### GPT-4 for Natural Language Processing
Powered by GPT-4,This model enhances the user experience by providing responses that are not only accurate but also contextually appropriate and insightful.

### Real-Time Insights and Analytics
Using Azure’s data processing capabilities, the project delivers real-time insights and visualizations.

### Other Feature 
- **Sales and Market Data Analysis**: Retrieve sales data and stock market information.
- **Predictive Analytics**: Forecast future sales using linear regression.
- **Data Visualization**: Generate revenue and profit charts for business insights.
- **Automated Reporting**: Generate PDF reports with visualizations.
- **Email Notification**: Send reports directly to specified recipients.


## Technologies Used

- **Azure AI Platform**: For deploying and managing the AI models and ensuring the application’s scalability and reliability.
- **GPT-4**: The latest language model by OpenAI, enabling natural and intuitive AI interactions.
- **Python Asyncio**: For managing asynchronous tasks, enhancing the application’s efficiency by handling multiple processes concurrently.
- **Data Handling and Transformation**:
  - `Pandas`: For data manipulation, enabling efficient handling of structured data.
  - `NumPy`: Used for numerical computations and array manipulation.
- **Visualization**:
  - `Matplotlib`: A fundamental library for creating static, animated, and interactive visualizations.
  - `Seaborn`: Built on Matplotlib, it provides high-level interface for attractive statistical graphics.
- **Finance API**:
  - `yfinance`: Fetches historical market data for stock analysis, aiding in predictive modeling and trend analysis.
- **Machine Learning**:
  - `sklearn.preprocessing.MinMaxScaler`: For scaling features to a specific range, essential in data normalization.
  - `sklearn.linear_model.LinearRegression`: Used to create predictive models, providing insights based on historical data.
- **PDF Generation**:
  - `FPDF`: Generates PDF reports, allowing users to save and share results in a professional format.
- **Email Notifications**:
  - `smtplib`, `MIMEText`, `MIMEMultipart`, `MIMEApplication`: Used to automate email notifications, attaching generated reports and sending them to designated recipients.
- **Environment Configuration**:
  - `dotenv`: Loads environment variables from `.env` file, keeping sensitive information secure.
  - **Custom Configuration**: `config` file imports (e.g., `AZURE_ENDPOINT`, `API_KEY`, `MODEL_NAME`) to access API keys and endpoints securely.
- **Agent-Based Communication**:
  - `autogen_agentchat.agents.AssistantAgent`, `autogen_agentchat.task.Console`, `autogen_agentchat.task.TextMentionTermination`, `autogen_agentchat.teams.RoundRobinGroupChat`: Facilitate AI-driven group chat features, enhancing collaboration and communication through AI.

- **Logging and Error Handling**:
  - `logging`: To track and manage errors, enabling robust debugging and maintenance.
  - `warnings`: For handling and filtering warnings in code execution.

---

## Sample Results

![Terminal Output 1](https://github.com/Mohamedzonkol/Business-Analysis-System/blob/main/Result/Output/Pasted%20image.png)
![Terminal Output 2](https://github.com/Mohamedzonkol/Business-Analysis-System/blob/main/Result/Output/Pasted%20image%20(3).png)
![Terminal Output 3](https://github.com/Mohamedzonkol/Business-Analysis-System/blob/main/Result/Output/Pasted%20image%20(2).png)

---
![Detailed Analysis Graph ](https://github.com/Mohamedzonkol/Business-Analysis-System/blob/main/Result/Analysis/Pasted%20image.png)


![Detailed Analysis Graph 2](https://github.com/Mohamedzonkol/Business-Analysis-System/blob/main/Result/Analysis/Pasted%20image%20(2).png)

---
## Project Output Files

The outputs, including PDF reports and analysis images, are available in the repository for review and download.

- [View PDF in GitHub Repository](https://github.com/Mohamedzonkol/Business-Analysis-System/tree/main/Result/Pdf)
