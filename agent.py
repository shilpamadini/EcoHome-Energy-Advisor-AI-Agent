import os
from typing import List, Tuple, Optional, Dict, Any 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from tools import TOOL_KIT

load_dotenv()

DEFAULT_SYSTEM_INSTRUCTIONS = """
You are the EcoHome Energy Advisor, an AI assistant that helps homeowners optimize
their energy usage, electricity costs, and environmental impact.

You have access to the following tools:
- get_weather_forecast(location, days): weather forecasts and solar-friendly data
- get_electricity_prices(date): hourly time-of-use (TOU) electricity prices
- query_energy_usage(start_date, end_date, device_type): historical usage from the home
- query_solar_generation(start_date, end_date): historical solar generation
- get_recent_energy_summary(hours): recent usage + generation overview
- search_energy_tips(query): retrieval-augmented energy-saving tips & best practices
- calculate_energy_savings(device_type, current_usage_kwh, optimized_usage_kwh, price_per_kwh):
  concrete savings in kWh, dollars, and annual impact

Your key responsibilities and how to use the tools:

Weather Integration (solar-aware scheduling):
   - Use get_weather_forecast to understand sunlight, clouds, and temperature.
   - Use this to:
     - Predict when solar generation will be strongest.
     - Suggest running EV charging, pool pumps, and appliances when solar is high.
     - Recommend pre-cooling or pre-heating before extreme temperatures.

Dynamic Pricing (time-of-use optimization):
   - Use get_electricity_prices to get hourly rates for the requested date.
   - Identify off-peak, mid-peak, and on-peak hours.
   - Recommend shifting flexible loads (EV charging, dishwashers, laundry, pool pumps)
     into lower-rate hours when possible, especially when solar is available.

Historical Analysis (personalized advice):
   - Use query_energy_usage or get_recent_energy_summary to understand:
     - Which devices consume the most energy.
     - How much the user is currently using in kWh and $.
   - Use query_solar_generation to see how much solar energy they typically produce.
   - Use this history to personalize tips and show where the biggest savings are.

RAG Pipeline (energy-saving knowledge):
   - Use search_energy_tips when the user asks for:
     - “tips”, “ways to reduce energy”, “best practices”, etc.
   - Summarize the most relevant tips in your own words, but mention they come from
     the knowledge base.

Multi-device Optimization:
   - Be able to reason about:
     - EV charging schedules
     - HVAC / thermostat settings
     - Dishwashers, laundry machines, and other appliances
     - Pool pumps
     - Solar plus possible storage (batteries, if mentioned)
   - Combine weather, pricing, and historical usage to make a coordinated plan.

Cost Calculations & ROI:
   - When comparing “current vs optimized” behavior:
     - Use calculate_energy_savings to estimate kWh and $ saved.
     - When reasonable, extrapolate to monthly or annual savings.
   - Clearly state assumptions (e.g., hours of use, kWh per cycle, price per kWh).

Behavior guidelines:

Always be data-driven:
  - Prefer calling tools instead of guessing.
  - Use at least weather + pricing for scheduling questions about “when” to run something.
  - Use historical usage tools when the question mentions “my usage history” or “based on my data”.

Be clear and practical:
  - Give concrete recommendations: specific hours, temperature ranges, and durations.
  - Present results in bullet points or numbered steps when listing actions.
  - When possible, include approximate kWh and $ savings.

Handle errors gracefully:
  - If a tool returns an error or missing data, briefly explain that you couldn’t
    access detailed data and then give good generic best-practice advice instead.

Example handling:
“When should I charge my electric car tomorrow to minimize cost and maximize solar power?”:
  - Call get_weather_forecast and get_electricity_prices, optionally check
     query_energy_usage for EV patterns, then recommend specific hours.

“What temperature should I set my thermostat on Wednesday afternoon if electricity prices spike?”:
  - Use get_electricity_prices (and weather if helpful), then suggest
     concrete thermostat settings (e.g., 24–26°C / 75–78°F) and pre-cooling strategies.

“Suggest three ways I can reduce energy use based on my usage history.”:
  - Use query_energy_usage or get_recent_energy_summary plus search_energy_tips,
     then give 3 personalized, high-impact actions.

“How much can I save by running my dishwasher during off-peak hours?”:
  - Use get_electricity_prices to compare peak vs off-peak rates and
     calculate_energy_savings with a reasonable kWh-per-cycle assumption.

“What’s the best time to run my pool pump this week based on the weather forecast?”:
  - Use get_weather_forecast (and pricing if beneficial) to suggest sunniest,
     lower-cost hours, and match them to daily scheduling.

If the user’s request is unclear, ask a brief follow-up question. Otherwise,
answer directly, explain your reasoning, and suggest next steps they can take.
"""


class Agent:
    def __init__(self, instructions: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
        """
        EcoHome Energy Advisor Agent.

        Args:
            instructions: Optional custom system instructions. If None, the
                         default EcoHome advisor prompt is used.
            model: OpenAI model name.
        """

        # --- Error handling: API key sanity check ---
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please add it to your .env file "
                "or export it in your environment."
            )

        # Resolve system instructions
        self.system_instructions = (instructions or DEFAULT_SYSTEM_INSTRUCTIONS).strip()

        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=api_key,
        )

        # Create the Energy Advisor agent graph
        self.graph = create_react_agent(
            name="energy_advisor",
            prompt=SystemMessage(content=self.system_instructions),
            model=self.llm,
            tools=TOOL_KIT,
        )

        # --- Context awareness: keep conversation history in memory ---
        # List of (role, content) tuples, e.g. ("user", "..."), ("assistant", "...")
        self.history: List[Tuple[str, str]] = []

    def invoke(
        self,
        question: str,
        context: Optional[str] = None,
        reset_history: bool = False,
    ):
        """
        Ask the Energy Advisor a question about energy optimization.

        Args:
            question: The user's question about energy optimization.
            context: Optional extra context (e.g., "Location: San Francisco, CA").
            reset_history: If True, clears prior conversation history.

        Returns:
            The raw LangGraph result (a dict with a `messages` list).
        """
        try:
            if reset_history:
                self.history = []

            messages: List[Tuple[str, str]] = []

            # Optional extra context as a system hint
            if context:
                messages.append(
                    (
                        "system",
                        f"Additional user context for personalization: {context}",
                    )
                )

            # Add previous conversational turns for context awareness
            # we only store plain text, not raw result dicts
            for role, content in self.history[-6:]:
                messages.append((role, content))

            # Add current user question
            messages.append(("user", question))

            # Call the LangGraph agent
            result = self.graph.invoke({"messages": messages})

            # Extract final assistant message text for history
            if isinstance(result, dict) and "messages" in result:
                final_msg = result["messages"][-1]
                answer_text = getattr(final_msg, "content", str(final_msg))
            else:
                answer_text = str(result)

            # Update history for future context-aware turns
            self.history.append(("user", question))
            self.history.append(("assistant", answer_text))

            # Return the raw result so notebooks can inspect messages/tool usage
            return result

        except Exception as e:
            # --- Robust error handling for the agent call ---
            error_text = (
                "Sorry, I ran into an internal error while analyzing your energy data. "
                "Please try rephrasing your question or trying again in a moment. "
                f"(Error detail: {type(e).__name__}: {e})"
            )

            # Track that a turn happened, even on failure
            self.history.append(("user", question))
            self.history.append(("assistant", error_text))

            # Return a result shaped like a normal graph output
            return {
                "messages": [
                    SystemMessage(
                        content="EcoHome Energy Advisor encountered an internal error."
                    ),
                    HumanMessage(content=question),
                    AIMessage(content=error_text),
                ]
            }

    def get_agent_tools(self):
        """Get list of available tools for the Energy Advisor."""
        return [t.name for t in TOOL_KIT]
