# EcoHome Energy Advisor

An AI-powered energy optimization agent that helps customers reduce electricity costs and environmental impact through personalized recommendations.

## Overview

EcoHome is a smart-home energy start-up that helps customers with solar panels, electric vehicles, and smart thermostats optimize their energy usage. The Energy Advisor agent provides personalized recommendations about when to run devices to minimize costs and carbon footprint.

### Key Features

- **Weather Integration**: Uses weather forecasts to predict solar generation
- **Dynamic Pricing**: Considers time-of-day electricity prices for cost optimization
- **Historical Analysis**: Queries past energy usage patterns for personalized advice
- **RAG Pipeline**: Retrieves relevant energy-saving tips and best practices
- **Multi-device Optimization**: Handles EVs, HVAC, appliances, and solar systems
- **Cost Calculations**: Provides specific savings estimates and ROI analysis

## Agent Capabilities

The EcoHome Energy Advisor intelligently analyzes weather, electricity prices, user history, and energy-efficiency knowledge to deliver personalized, actionable recommendations for smart homes.

### 1. Real-Time Weather & Solar Forecasting

Fetches live weather data from the OpenWeather API

Includes hourly temperature, cloud cover, and solar irradiance

Predicts when rooftop solar will generate the most power

Uses weather insight to shift appliance or EV usage to sunny periods

### 2. Smart Scheduling & Optimization

Determines the best time to run appliances and devices

Minimizes electricity cost and grid dependence

Avoids peak-pricing windows automatically

### 3. Time-of-Use Pricing Intelligence

Retrieves daily electricity rate schedules

Identifies peak, mid-peak, & off-peak hours

Calculates cost differences for scheduling decisions

### 4. Personalized Energy Insights

Analyzes historical household consumption

Identifies energy-intensive appliances

Provides tailored cost-saving recommendations

### 5. RAG-Powered Energy Recommendations

Retrieves stored sustainability and efficiency tips

Summarizes best practices based on user context

### 6. Cost & Savings Estimation

Computes projected energy costs

Estimates monthly/annual savings from schedule shifts

Supports ROI analysis for behavioral changes

### 7. Multi-Device Optimization

Optimizes usage for:

EV charging

HVAC & thermostats

Dishwashers, dryers, washing machines

Pool pumps

Solar systems & home batteries

### 8. Context-Aware, Actionable Responses

Provides concrete time windows, pricing, and reasoning

Communicates steps clearly and concisely

Adapts based on previous conversation history

### 9. Safe & Robust Tool Handling

Only uses tools when needed

Handles API errors gracefully

Falls back to general best practices when data unavailable

### Tools Available

- **Weather Forecast**: Get hourly weather predictions and solar irradiance
- **Electricity Pricing**: Access time-of-day pricing data
- **Energy Usage Query**: Retrieve historical consumption data
- **Solar Generation Query**: Get past solar production data
- **Energy Tips Search**: Find relevant energy-saving recommendations
- **Savings Calculator**: Compute potential cost savings

### Example Questions

The Energy Advisor can answer questions like:

- "When should I charge my electric car tomorrow to minimize cost and maximize solar power?"
- "What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?"
- "Suggest three ways I can reduce energy use based on my usage history."
- "How much can I save by running my dishwasher during off-peak hours?"

## Database Schema

### Energy Usage Table
- `timestamp`: When the energy was consumed
- `consumption_kwh`: Amount of energy used
- `device_type`: Type of device (EV, HVAC, appliance)
- `device_name`: Specific device name
- `cost_usd`: Cost at time of usage

### Solar Generation Table
- `timestamp`: When the energy was generated
- `generation_kwh`: Amount of solar energy produced
- `weather_condition`: Weather during generation
- `temperature_c`: Temperature at time of generation
- `solar_irradiance`: Solar irradiance level


## Key Technologies Used

- **LangChain**: Agent framework and tool integration
- **LangGraph**: Agent orchestration and workflow
- **ChromaDB**: Vector database for document retrieval
- **SQLAlchemy**: Database ORM and management
- **OpenAI**: LLM and embeddings
- **SQLite**: Local database storage

## Evaluation Criteria

The agent is evaluated on:

- **Accuracy**: Correct information and calculations
- **Relevance**: Responses address the user's question
- **Completeness**: Comprehensive answers with actionable advice
- **Tool Usage**: Appropriate use of available tools
- **Reasoning**: Clear explanation of recommendations

## Project Structure

```
ecohome/
├── models/
│   ├── __init__.py
│   └── energy.py                     # Database models for energy data
├── data/
│   └── documents/                    # RAG knowledge base
│       ├── tip_device_best_practices.txt
│       ├── tip_energy_savings.txt
│       ├── tip_solar_panel_efficiency.txt
│       ├── tip_ev_charging_strategies.txt
│       ├── tip_hvac_optimization.txt
│       ├── tip_pool_pump_best_practices.txt
│       └── tip_time_of_use_planning.txt
├── agent.py                          # Main Energy Advisor agent
├── tools.py                          # Agent tools (weather, pricing, database, RAG)
├── requirements.txt                  # Python dependencies
├── 01_db_setup.ipynb                 # Database setup and sample data
├── 02_rag_setup.ipynb                # RAG pipeline setup
├── 03_agent_evaluation.ipynb         # Agent testing and evaluation
├── 04_agent_run.ipynb                # Running the agent with examples
└── README.md                          # Project documentation

```

## Setup Instructions

### 1. Clone the Repository
```
git clone https://github.com/shilpamadini/ecohome-energy-advisor.git
cd ecohome-energy-advisor
```
### 2. create conda environment

```conda create -n ecohome python=3.10 -y
```

### 3. Activate conda environment

```conda activate ecohome
```

### 4. Install Dependencies

```
conda install pip -y
```
```
pip install -r requirements.txt
```
### 5. Configure Environment Variables

```touch .env
```
Edit the .env to add keys for openai and openweathermap ai

```
OPENAI_API_KEY=your_openai_key_here
OPENWEATHERMAP_API_KEY=your_weather_api_key_here
```

### 6. Add Environment for Jupyter Lab

```
pip install ipykernel
python -m ipykernel install --user --name ecohome
```


### 7. Run the Notebooks

Execute the notebooks in order:

1. **01_db_setup.ipynb** - Set up the database and populate with sample data
2. **02_rag_setup.ipynb** - Configure the RAG pipeline for energy tips
3. **03_agent_evaluation.ipynb** - Test and evaluate the agent
4. **04_agent_run.ipynb** - Run the agent with example scenarios

