import asyncio
from typing import Dict, List
import json
from datetime import datetime
import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.task import Console, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import AzureOpenAIChatCompletionClient
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import logging
import os
from typing import Optional, Union
import warnings
 from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='business_analysis.log'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

Load environment variables
load_dotenv()

class BusinessAnalytics:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = LinearRegression()

    async def query_sales_data(self) -> Dict:
        try:
            return {
                "total_sales": 1500000,
                "sales_by_product": {
                    "Product A": {"revenue": 500000, "units": 1000, "cost": 300000},
                    "Product B": {"revenue": 450000, "units": 900, "cost": 270000},
                    "Product C": {"revenue": 350000, "units": 700, "cost": 210000}
                },
                "sales_by_region": {
                    "North": {"revenue": 500000, "growth": 15.5},
                    "South": {"revenue": 450000, "growth": 12.3},
                    "East": {"revenue": 350000, "growth": 8.7},
                    "West": {"revenue": 200000, "growth": 5.2}
                },
                "monthly_growth": 15.5,
                "year_over_year_growth": 25.3,
                "profit_margin": 35.8
            }
        except Exception as e:
            logger.error(f"Error querying sales data: {str(e)}")
            raise

    async def get_market_data(self, symbol: str = "AAPL") -> Dict:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            return {
                "current_price": hist['Close'][-1],
                "yearly_high": hist['High'].max(),
                "yearly_low": hist['Low'].min(),
                "volume": hist['Volume'][-1],
                "market_cap": stock.info['marketCap']
            }
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            raise




    async def predict_future_sales(self, historical_data: Optional[Dict] = None) -> Dict:
        try:
            if historical_data is None:
                historical_data = await self.query_sales_data()

            # Create a simple prediction model using the sales_by_product data
            sales_data = pd.DataFrame.from_dict(
                historical_data['sales_by_product'],
                orient='index'
            )

            X = np.array(sales_data['units']).reshape(-1, 1)
            y = np.array(sales_data['revenue'])
            
            self.model.fit(X, y)
            
            # Predict for different future scenarios
            future_units = np.array([[800], [1000], [1200], [1400]])
            predicted_revenues = self.model.predict(future_units)

            predictions = {
                f"scenario_{i+1}": {
                    "units": int(units[0]),
                    "predicted_revenue": float(revenue),
                }
                for i, (units, revenue) in enumerate(zip(future_units, predicted_revenues))
            }

            return {
                "predictions": predictions,
                "confidence_score": float(self.model.score(X, y)),
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "model_coefficients": {
                    "slope": float(self.model.coef_[0]),
                    "intercept": float(self.model.intercept_)
                }
            }

        except Exception as e:
            logger.error(f"Error in sales prediction: {str(e)}")
            return {
                "error": str(e),
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "status": "failed"
            }




    async def generate_visualizations(self, data: Dict) -> List[str]:
        try:
            charts = []
            
            # Sales by Region
            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(data['sales_by_region'].keys()),
                    y=[d['revenue'] for d in data['sales_by_region'].values()])
            plt.title('Sales by Region')
            plt.xlabel('Region')
            plt.ylabel('Revenue ($)')
            plt.savefig('sales_by_region.png')
            charts.append('sales_by_region.png')
            
            # Product Performance
            plt.figure(figsize=(12, 6))
            product_data = data['sales_by_product']
            profit = {k: v['revenue'] - v['cost'] for k, v in product_data.items()}
            plt.pie(profit.values(), labels=profit.keys(), autopct='%1.1f%%')
            plt.title('Product Profit Distribution')
            plt.savefig('product_profit.png')
            charts.append('product_profit.png')
            
            return charts
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise



    async def generate_pdf_report(self, content: Dict, charts: List[str]) -> str:
        try:
            # Ensure content includes the necessary data
            if 'sales_data' in content:
                data = content['sales_data']
            else:
                data = await self.query_sales_data()

            pdf = FPDF()
            pdf.add_page()
            
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Business Analysis Report', ln=True, align='C')
            pdf.ln(10)
            
            pdf.set_font('Arial', '', 12)
            for section, text in content.items():
                if section == 'sales_data':
                    for key, value in text.items():
                        pdf.set_font('Arial', 'B', 14)
                        pdf.cell(0, 10, key, ln=True)
                        pdf.set_font('Arial', '', 12)
                        pdf.multi_cell(0, 10, str(value))
                        pdf.ln(5)
                else:
                    pdf.set_font('Arial', 'B', 14)
                    pdf.cell(0, 10, section, ln=True)
                    pdf.set_font('Arial', '', 12)
                    pdf.multi_cell(0, 10, str(text))
                    pdf.ln(5)
        
            # Add charts
            for chart in charts:
                pdf.add_page()
                pdf.image(chart, x=10, y=10, w=190)

            filename = f"business_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            pdf.output(filename)
            return filename
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise
    # async def generate_pdf_report(self, content: Union[Dict, str], charts: Optional[List[str]] = None) -> str:
    #     try:
    #         pdf = FPDF()
    #         pdf.add_page()
            
    #         pdf.set_font('Arial', 'B', 20)
    #         pdf.set_text_color(0, 51, 102)  # Dark blue color
    #         pdf.cell(0, 15, 'Business Analysis Report', ln=True, align='C')
    #         pdf.line(10, pdf.get_y(), 200, pdf.get_y())  # Add a horizontal line
    #         pdf.ln(10)

    #         pdf.set_font('Arial', 'I', 10)
    #         pdf.set_text_color(128, 128, 128)  # Gray color
    #         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         pdf.cell(0, 10, f'Generated on: {timestamp}', ln=True, align='R')
    #         pdf.ln(5)

    #         if isinstance(content, dict):
    #             for section, text in content.items():
    #                 pdf.set_font('Arial', 'B', 14)
    #                 pdf.set_text_color(0, 102, 204)  # Blue color for headers
    #                 pdf.cell(0, 10, str(section), ln=True)
                    
    #                 pdf.set_font('Arial', '', 12)
    #                 pdf.set_text_color(0, 0, 0)  # Black color for content
    #                 if isinstance(text, (dict, list)):
    #                     text_content = json.dumps(text, indent=2)
    #                 else:
    #                     text_content = str(text)

    #                 for line in text_content.split('\n'):
    #                     try:
    #                         pdf.cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
    #                     except Exception:
    #                         pdf.cell(0, 8, line.encode('latin-1', 'ignore').decode('latin-1'))
    #                 pdf.ln(5)
    #         else:
    #             pdf.set_font('Arial', '', 12)
    #             pdf.set_text_color(0, 0, 0)
    #             try:
    #                 pdf.multi_cell(0, 8, content.encode('latin-1', 'replace').decode('latin-1'))
    #             except Exception:
    #                 pdf.multi_cell(0, 8, content.encode('latin-1', 'ignore').decode('latin-1'))

    #         if charts:
    #             for chart in charts:
    #                 if os.path.exists(chart):
    #                     pdf.add_page()
    #                     try:
    #                         pdf.image(chart, x=10, y=10, w=190)
    #                     except Exception as e:
    #                         logger.error(f"Error adding chart {chart}: {str(e)}")
    #                         pdf.cell(0, 10, f"Error loading chart: {chart}", ln=True)

    #         pdf.set_auto_page_break(auto=True, margin=15)
    #         pdf.set_y(-15)
    #         pdf.set_font('Arial', 'I', 8)
    #         pdf.set_text_color(128, 128, 128)
    #         pdf.cell(0, 10, f'Page {pdf.page_no()}', align='C')

    #         filename = f"business_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    #         pdf.output(filename)
    #         logger.info(f"Successfully generated PDF report: {filename}")
    #         return filename

    #     except Exception as e:
    #         error_msg = f"Error generating PDF report: {str(e)}"
    #         logger.error(error_msg)
    #         raise Exception(error_msg)

    async def send_email_report(self, report_file: str, recipient: str) -> bool:
        try:
            smtp_server = os.getenv('SMTP_SERVER')
            smtp_port = int(os.getenv('SMTP_PORT'))
            sender_email = os.getenv('SENDER_EMAIL')
            sender_password = os.getenv('SENDER_PASSWORD')

            msg = MIMEMultipart()
            msg['Subject'] = 'Business Analysis Report'
            msg['From'] = sender_email
            msg['To'] = recipient

            body = "Please find attached the latest business analysis report."
            msg.attach(MIMEText(body, 'plain'))

            with open(report_file, 'rb') as f:
                attachment = MIMEApplication(f.read(), _subtype='pdf')
                attachment.add_header('Content-Disposition', 'attachment', filename=report_file)
                msg.attach(attachment)

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            return True
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False

async def main():
    try:
        analytics = BusinessAnalytics()

        azure_endpoint = "https://20812-m3elzq48-australiaeast.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview/"
        api_key = "3RpEWhxq4STdlhD4tuuPstRd7sZxIJEcMF45yhMcdSQGS0y8yjP3JQQJ99AKACL93NaXJ3w3AAABACOGqbKm"
        model_name = "gpt-4"

        try:
            model_client = AzureOpenAIChatCompletionClient(
                model=model_name,
                api_version="2024-08-01-preview",
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                model_capabilities={
                    "vision": True,
                    "function_calling": True,
                    "json_output": True,
                }
            )
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {str(e)}")
            raise

        agents = []
        try:
            data_analyst = AssistantAgent(
                name="data_analyst",
                model_client=model_client,
                tools=[
                    analytics.query_sales_data,
                    analytics.get_market_data,
                    analytics.predict_future_sales
                ],
                system_message="""Advanced data analyst with expertise in:
                - Complex data analysis and visualization
                - Market trend analysis
                - Predictive modeling
                - Financial metrics interpretation"""
            )
            agents.append(data_analyst)

            business_analyst = AssistantAgent(
                name="business_analyst",
                model_client=model_client,
                tools=[],
                system_message="""Strategic business analyst focusing on:
                - Market opportunity identification
                - Competitive analysis
                - Risk assessment
                - Growth strategy development"""
            )
            agents.append(business_analyst)

            report_generator = AssistantAgent(
                name="report_generator",
                model_client=model_client,
                tools=[
                    analytics.generate_visualizations,
                    analytics.generate_pdf_report
                ],
                system_message="""Report generation specialist handling:
                - Data visualization
                - PDF report creation
                - Executive summary preparation
                - Professional report formatting"""
            )
            agents.append(report_generator)

        except Exception as e:
            logger.error(f"Error creating agents: {str(e)}")
            raise

        termination = TextMentionTermination("TERMINATE")
        agent_team = RoundRobinGroupChat(agents, termination_condition=termination)

        analysis_task = """
        Perform a comprehensive business analysis with the following steps:
        
        1. Data Collection and Analysis:
           - Gather current sales performance data
           - Analyze market trends and patterns
           - Evaluate customer feedback and metrics
        
        2. Market Analysis:
           - Analyze competitive landscape
           - Identify market opportunities
           - Assess market risks and challenges
        
        3. Predictive Analysis:
           - Generate future sales predictions
           - Identify growth patterns
           - Assess market potential
        
        4. Visualization and Reporting:
           - Create clear and informative visualizations
           - Generate comprehensive PDF report
           - Include executive summary and recommendations
        
        Please provide detailed insights and recommendations based on the analysis.
        Once complete, generate a professional PDF report including all visualizations and findings.
        """

        sales_data = await analytics.query_sales_data()

        stream = agent_team.run_stream(task=analysis_task, initial_context={"sales_data": sales_data})
        
        
        await Console(stream)


    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

