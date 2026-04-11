import gradio as gr
import os
import base64
from io import BytesIO
from PIL import Image
import requests
import json
import re
from datetime import datetime

# --- Configuration ---

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Models
PERPLEXITY_MODEL = "sonar-pro"
OPENAI_VISION_MODEL = "gpt-5.2"

# Currency mapping
CURRENCY_MAP = {
    "US": "USD", "USA": "USD", "United States": "USD",
    "CA": "CAD", "Canada": "CAD",
    "MX": "MXN", "Mexico": "MXN",
    "UK": "GBP", "GB": "GBP", "United Kingdom": "GBP", "England": "GBP", "London": "GBP",
    "DE": "EUR", "Germany": "EUR", "Berlin": "EUR", "Munich": "EUR","Frankfurt": "EUR",
    "FR": "EUR", "France": "EUR", "Paris": "EUR",
    "IT": "EUR", "Italy": "EUR", "Milan": "EUR", "Rome": "EUR",
    "ES": "EUR", "Spain": "EUR", "Madrid": "EUR", "Barcelona": "EUR",
    "NL": "EUR", "Netherlands": "EUR", "Amsterdam": "EUR",
    "BE": "EUR", "Belgium": "EUR", "Brussels": "EUR",
    "AT": "EUR", "Austria": "EUR", "Vienna": "EUR",
    "IE": "EUR", "Ireland": "EUR", "Dublin": "EUR",
    "PT": "EUR", "Portugal": "EUR", "Lisbon": "EUR",
    "GR": "EUR", "Greece": "EUR", "Athens": "EUR",
    "CH": "CHF", "Switzerland": "CHF", "Zurich": "CHF",
    "JP": "JPY", "Japan": "JPY", "Tokyo": "JPY",
    "CN": "CNY", "China": "CNY", "Beijing": "CNY", "Shanghai": "CNY",
    "IN": "INR", "India": "INR", "Mumbai": "INR", "Delhi": "INR",
    "AU": "AUD", "Australia": "AUD", "Sydney": "AUD", "Melbourne": "AUD",
    "NZ": "NZD", "New Zealand": "NZD", "Auckland": "NZD",
    "SG": "SGD", "Singapore": "SGD",
    "HK": "HKD", "Hong Kong": "HKD",
    "KR": "KRW", "Korea": "KRW", "South Korea": "KRW", "Seoul": "KRW",
    "TH": "THB", "Thailand": "THB", "Bangkok": "THB",
    "MY": "MYR", "Malaysia": "MYR", "Kuala Lumpur": "MYR",
    "AE": "AED", "UAE": "AED", "Dubai": "AED", "Abu Dhabi": "AED",
    "SA": "SAR", "Saudi Arabia": "SAR", "Riyadh": "SAR",
    "IL": "ILS", "Israel": "ILS", "Tel Aviv": "ILS",
    "SE": "SEK", "Sweden": "SEK", "Stockholm": "SEK",
    "NO": "NOK", "Norway": "NOK", "Oslo": "NOK",
    "DK": "DKK", "Denmark": "DKK", "Copenhagen": "DKK",
    "BR": "BRL", "Brazil": "BRL", "Sao Paulo": "BRL",
    "ZA": "ZAR", "South Africa": "ZAR", "Johannesburg": "ZAR",
}

CURRENCY_SYMBOLS = {
    "USD": "$", "CAD": "CA$", "MXN": "MX$",
    "GBP": "£", "EUR": "€",
    "JPY": "¥", "CNY": "¥", "INR": "₹",
    "AUD": "A$", "NZD": "NZ$", "SGD": "S$",
    "HKD": "HK$", "KRW": "₩", "THB": "฿",
    "MYR": "RM", "AED": "د.إ", "SAR": "﷼",
    "ILS": "₪", "CHF": "CHF", "SEK": "kr",
    "NOK": "kr", "DKK": "kr", "BRL": "R$",
    "ZAR": "R"
}

# Repair philosophies
REPAIR_PHILOSOPHIES = {
    "primary": {
        "name": "OEM Parts - Dealer Service",
        "description": "Using Original Equipment Manufacturer (OEM) parts at authorized dealer service centers",
        "parts_type": "OEM (Original Equipment Manufacturer)",
        "labor_source": "Authorized Dealer",
        "quality_level": "Premium",
        "warranty": "Manufacturer warranty included"
    },
    "alternative": {
        "name": "Aftermarket Parts - Independent Shop",
        "description": "Using quality aftermarket parts at certified independent repair shops",
        "parts_type": "Aftermarket (Quality Certified)",
        "labor_source": "Independent Certified Shop",
        "quality_level": "Standard/Good",
        "warranty": "Shop warranty (typically 12 months)"
    }
}

# --- Agent State Management ---

def create_agent_state(agent_type, config=None):
    """Create initial state for an agent"""
    return {
        'type': agent_type,
        'memory': [],
        'confidence': 1.0,
        'last_action': None,
        'timestamp': datetime.now().isoformat(),
        'config': config or {},
        'performance_history': [],
        'react_trace': []  
    }

def update_agent_memory(agent_state, event):
    """Update agent's memory with new event"""
    agent_state['memory'].append({
        'timestamp': datetime.now().isoformat(),
        'event': event
    })
    # Keep only last 10 events
    if len(agent_state['memory']) > 10:
        agent_state['memory'] = agent_state['memory'][-10:]
    return agent_state

def agent_decide(agent_state, context):
    """Agent makes decision based on state and context"""
    decision = {
        'action': None,
        'confidence': agent_state['confidence'],
        'reasoning': []
    }

    # Decision logic based on agent type and context
    if agent_state['type'] == 'vision':
        decision['action'] = 'analyze_image'
        decision['reasoning'].append('Image received, proceeding with analysis')

    elif agent_state['type'] == 'cost_estimator':
        severity = context.get('severity', 'moderate')
        if severity == 'severe':
            decision['action'] = 'detailed_estimate'
            decision['reasoning'].append('Severe damage requires detailed cost breakdown')
        else:
            decision['action'] = 'standard_estimate'
            decision['reasoning'].append('Standard estimation approach')

    elif agent_state['type'] == 'shop_finder':
        location = context.get('location')
        if location:
            decision['action'] = 'search_shops'
            decision['reasoning'].append(f'Location provided: {location}')
        else:
            decision['action'] = 'skip'
            decision['reasoning'].append('No location provided')

    return decision

# --- ReAct Loop ---

def react_thought(agent_state, thought_text):
    """Record a Thought step in the ReAct loop"""
    step = {
        'step': 'Thought',
        'content': thought_text,
        'timestamp': datetime.now().isoformat()
    }
    agent_state['react_trace'].append(step)
    return agent_state

def react_action(agent_state, action_name, action_input=None):
    """Record an Action step in the ReAct loop"""
    step = {
        'step': 'Action',
        'content': action_name,
        'input': action_input,
        'timestamp': datetime.now().isoformat()
    }
    agent_state['react_trace'].append(step)
    return agent_state

def react_observation(agent_state, observation_text):
    """Record an Observation step in the ReAct loop"""
    step = {
        'step': 'Observation',
        'content': observation_text,
        'timestamp': datetime.now().isoformat()
    }
    agent_state['react_trace'].append(step)
    return agent_state

def react_run_loop(agent_state, thought_text, action_name, action_fn, action_input=None):
    """
    Execute a single Thought → Action → Observation ReAct iteration.
    - thought_text : reasoning string before acting
    - action_name  : label for the action being taken
    - action_fn    : callable that performs the action; receives action_input, returns (result, observation_text)
    - action_input : optional data passed to action_fn
    Returns (agent_state, result)
    """
    # Thought
    agent_state = react_thought(agent_state, thought_text)

    # Action
    agent_state = react_action(agent_state, action_name, action_input)

    # Execute and get Observation
    result, observation_text = action_fn(action_input)

    # Observation
    agent_state = react_observation(agent_state, observation_text)

    return agent_state, result

# --- Helper Functions ---

def detect_currency_from_location(location):
    """Detect currency from location"""
    if not location:
        return "EUR", "€"

    location_clean = location.strip()
    for key, currency in CURRENCY_MAP.items():
        if key.lower() in location_clean.lower():
            return currency, CURRENCY_SYMBOLS.get(currency, currency)

    return "EUR", "€"

def image_to_base64(image):
    """Convert PIL Image to base64"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# --- Vision Agent ---

def create_vision_agent(api_key):
    """Create a vision analysis agent"""
    agent_state = create_agent_state('vision', {
        'api_key': api_key,
        'confidence_threshold': 0.7
    })
    return agent_state

def vision_agent_perceive(agent_state, image):
    """Vision agent perceives and analyzes the image using a ReAct loop"""
    decision = agent_decide(agent_state, {'has_image': image is not None})

    if decision['action'] != 'analyze_image':
        return agent_state, None

    headers = {
        "Authorization": f"Bearer {agent_state['config']['api_key']}",
        "Content-Type": "application/json"
    }

    img_base64 = image_to_base64(image)

    # --- ReAct loop ---
    def call_vision_api(payload):
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        raw = result['choices'][0]['message']['content'].strip()

        # Strip markdown fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()

        # Parse structured JSON returned by the model
        vision_data = None
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                vision_data = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        if vision_data is None:
            try:
                vision_data = json.loads(raw)
            except json.JSONDecodeError:
                vision_data = {}

        # Normalise severity — model returns exactly one of: minor / moderate / severe
        severity = vision_data.get('severity', 'moderate').strip().lower()
        if severity not in ('minor', 'moderate', 'severe'):
            severity = 'moderate'

        description = vision_data.get('description', raw)
        confidence = float(vision_data.get('confidence', 0.9))

        result_data = {
            'description': description,
            'severity': severity,
            'confidence': confidence,
            'damaged_parts': vision_data.get('damaged_parts', []),
            'damage_type': vision_data.get('damage_type', ''),
            'observations': vision_data.get('observations', '')
        }
        observation = (
            f"Vision API responded successfully. "
            f"Detected severity: {severity}. "
            f"Confidence: {confidence:.0%}."
        )
        return result_data, observation

    api_payload = {
        "model": OPENAI_VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert auto damage evaluator. "
                    "You always respond with a single valid JSON object and nothing else — "
                    "no prose, no markdown fences."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyze this car damage image and respond ONLY with this JSON structure:
{
  "description": "Detailed technical description of all visible damage",
  "damaged_parts": ["list", "of", "specific", "part", "names"],
  "damage_type": "e.g. rust, scratch, dent, collision, paint peel",
  "severity": "minor OR moderate OR severe",
  "confidence": 0.0 to 1.0,
  "observations": "Any additional relevant observations"
}

Severity scale:
- minor: surface scratches, small chips, light scuffs — no structural impact
- moderate: dents, localised rust, cracked bumper — panel repair needed
- severe: deep rust perforation, major collision deformation, frame damage

Reply with JSON only."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    }
                ]
            }
        ],
        "max_completion_tokens": 600
    }

    agent_state, result_data = react_run_loop(
        agent_state,
        thought_text=(
            "I have received a car damage image. "
            "I need to call the vision model to identify damaged parts, "
            "assess the severity, and extract a structured description."
        ),
        action_name="call_openai_vision_api",
        action_fn=call_vision_api,
        action_input=api_payload
    )

    # Guard against None result
    if not result_data:
        result_data = {'description': 'Vision analysis unavailable.', 'severity': 'moderate', 'confidence': 0.0}

    # Update agent state
    agent_state = update_agent_memory(agent_state, {
        'action': 'image_analyzed',
        'severity_detected': result_data['severity'],
        'confidence': result_data.get('confidence', 0.9)
    })

    return agent_state, result_data

# --- Cost Estimation Agent ---

def create_cost_agent(api_key, philosophy):
    """Create a cost estimation agent with specific repair philosophy"""
    agent_state = create_agent_state('cost_estimator', {
        'api_key': api_key,
        'philosophy': philosophy,
        'price_adjustment_factor': 1.0
    })
    return agent_state

def cost_agent_estimate(agent_state, damage_info, location, currency):
    """Cost agent generates repair estimate using a ReAct loop"""
    # Guard: damage_info may be None if vision agent failed
    if not damage_info:
        damage_info = {'description': 'Car damage visible in uploaded image.', 'severity': 'moderate'}

    decision = agent_decide(agent_state, {
        'severity': damage_info.get('severity', 'moderate')
    })

    philosophy_key = agent_state['config']['philosophy']
    philosophy = REPAIR_PHILOSOPHIES[philosophy_key]
    currency_symbol = CURRENCY_SYMBOLS.get(currency, currency)

    headers = {
        "Authorization": f"Bearer {agent_state['config']['api_key']}",
        "Content-Type": "application/json"
    }

    damage_description = damage_info.get('description', 'Car damage visible in uploaded image.')
    damage_severity = damage_info.get('severity', 'moderate')

    prompt = f"""Based on this car damage, provide repair cost estimate for {location} using {philosophy['description']}:

Damage: {damage_description}
Severity: {damage_severity}

Repair Philosophy:
- Parts: {philosophy['parts_type']}
- Service: {philosophy['labor_source']}
- Quality: {philosophy['quality_level']}

Provide JSON response:
{{
    "damage_summary": "Brief technical summary of the damage",
    "affected_parts": ["list of specific car parts damaged"],
    "estimated_cost_low": numeric value,
    "estimated_cost_high": numeric value,
    "currency": "{currency}",
    "cost_breakdown": "Detailed breakdown in NATURAL LANGUAGE explaining parts costs, labor hours and rates, painting, and any other costs. Mention {philosophy['parts_type']} and {philosophy['labor_source']} with specific prices and hourly rates. Write as a flowing paragraph, NOT as JSON or dictionary.",
    "repair_philosophy": "{philosophy['name']}",
    "warranty_info": "{philosophy['warranty']}",
    "sources": ["source descriptions WITHOUT citation numbers - plain text like 'RepairPal.com costs' or 'Local labor rates'"]
}}

CRITICAL: The "cost_breakdown" field must be a NATURAL LANGUAGE PARAGRAPH, not a nested JSON object or dictionary. Write it as readable text with specific costs mentioned inline.

Requirements:
- Costs in {currency} ({currency_symbol})
- Based on {philosophy['name']} in {location}
- ONLY JSON, no markdown"""

    api_payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in auto repair cost estimation. Provide accurate estimates based on repair philosophies. List sources as plain text without citation numbers."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    # --- ReAct loop ---
    def call_cost_api(payload):
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        response_text = result['choices'][0]['message']['content'].strip()

        # Strip markdown code fences if present
        response_text = re.sub(r'^```(?:json)?\s*', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'\s*```$', '', response_text, flags=re.MULTILINE)
        response_text = response_text.strip()

        # Extract JSON object
        cost_data = None
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                cost_data = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        if cost_data is None:
            try:
                cost_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback skeleton so the app never hard-crashes
                cost_data = {
                    'damage_summary': damage_description,
                    'affected_parts': ['Unable to parse'],
                    'estimated_cost_low': 0,
                    'estimated_cost_high': 0,
                    'currency': currency,
                    'cost_breakdown': f'Cost estimation unavailable. Raw response: {response_text[:300]}',
                    'repair_philosophy': philosophy['name'],
                    'warranty_info': philosophy['warranty'],
                    'sources': []
                }

        # Post-process: Convert dictionary cost_breakdown to natural language if needed
        if isinstance(cost_data.get('cost_breakdown'), dict):
            breakdown_dict = cost_data['cost_breakdown']
            breakdown_text = "Costs include: "
            parts_info = []
            labor_info = []
            for key, value in breakdown_dict.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if 'part' in subkey.lower() or 'material' in subkey.lower():
                            parts_info.append(f"{subkey.replace('_', ' ')} ({subvalue} {currency})")
                        elif 'labor' in subkey.lower() or 'hour' in subkey.lower():
                            labor_info.append(f"{subkey.replace('_', ' ')} ({subvalue})")
                elif isinstance(value, (int, float)):
                    parts_info.append(f"{key.replace('_', ' ')} ({value} {currency})")
            if parts_info:
                breakdown_text += ", ".join(parts_info[:5])
            if labor_info:
                breakdown_text += "; Labor: " + ", ".join(labor_info[:3])
            cost_data['cost_breakdown'] = (
                breakdown_text if len(breakdown_text) > 20
                else f"Estimate includes {philosophy['parts_type']} and {philosophy['labor_source']} labor rates for {location}."
            )

        low = cost_data.get('estimated_cost_low', 0)
        high = cost_data.get('estimated_cost_high', 0)
        observation = (
            f"Cost API ({philosophy_key}) responded successfully. "
            f"Estimated range: {currency_symbol}{low:,.0f} - {currency_symbol}{high:,.0f} {currency}."
        )
        return cost_data, observation

    agent_state, cost_data = react_run_loop(
        agent_state,
        thought_text=(
            f"I have damage information with severity '{damage_severity}' "
            f"and I need to generate a {decision['action']} for the '{philosophy_key}' repair philosophy "
            f"in {location} using {currency}. "
            f"I will query the Perplexity API with structured requirements."
        ),
        action_name="call_perplexity_cost_api",
        action_fn=call_cost_api,
        action_input=api_payload
    )

    # Update agent memory
    agent_state = update_agent_memory(agent_state, {
        'action': 'estimate_generated',
        'philosophy': philosophy_key,
        'cost_range': f"{cost_data.get('estimated_cost_low')}-{cost_data.get('estimated_cost_high')}"
    })

    return agent_state, cost_data

# --- Shop Finder Agent ---

def create_shop_finder_agent(api_key):
    """Create shop finder agent"""
    agent_state = create_agent_state('shop_finder', {
        'api_key': api_key,
        'search_radius': 'default'
    })
    return agent_state

def shop_finder_search(agent_state, location):
    """Shop finder agent searches for repair shops using a ReAct loop"""
    decision = agent_decide(agent_state, {'location': location})

    if decision['action'] == 'skip':
        return agent_state, "Enter a location to find nearby repair shops."

    headers = {
        "Authorization": f"Bearer {agent_state['config']['api_key']}",
        "Content-Type": "application/json"
    }

    api_payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You find local auto body shops using current search results."
            },
            {
                "role": "user",
                "content": f"""Find 3-5 highly-rated auto body shops near {location} for collision repair.

For each shop provide:
- Shop Name
- Address
- Phone number
- Website link
- Rating
- Brief description of services and specialties

Format clearly with sections."""
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.2
    }

    # --- ReAct loop ---
    def call_shop_api(payload):
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        shops_text = result['choices'][0]['message']['content']
        shop_count = shops_text.count('\n\n')
        observation = (
            f"Shop search API responded successfully for location '{location}'. "
            f"Retrieved shop listings ({shop_count} sections found in response)."
        )
        return shops_text, observation

    agent_state, shops_text = react_run_loop(
        agent_state,
        thought_text=(
            f"I need to find auto body repair shops near '{location}'. "
            f"I will query Perplexity with a structured request for shop names, "
            f"addresses, ratings, and contact details."
        ),
        action_name="call_perplexity_shop_search_api",
        action_fn=call_shop_api,
        action_input=api_payload
    )

    agent_state = update_agent_memory(agent_state, {
        'action': 'shops_found',
        'location': location
    })

    return agent_state, shops_text

# --- Orchestrator Agent ---

def create_orchestrator():
    """Create orchestrator agent that coordinates all other agents"""
    agent_state = create_agent_state('orchestrator', {
        'execution_plan': [],
        'agent_registry': {}
    })
    return agent_state

def orchestrator_plan(orchestrator_state, inputs):
    """Orchestrator creates execution plan based on inputs using a ReAct loop"""

    def build_plan(inp):
        plan = []
        if inp.get('image'):
            plan.append({
                'agent': 'vision',
                'priority': 1,
                'reason': 'Image analysis required'
            })
        if inp.get('location'):
            plan.append({
                'agent': 'cost_primary',
                'priority': 2,
                'reason': 'Primary quote needed'
            })
            plan.append({
                'agent': 'cost_alternative',
                'priority': 2,
                'reason': 'Alternative quote for comparison'
            })
            plan.append({
                'agent': 'shop_finder',
                'priority': 3,
                'reason': 'Find local repair shops'
            })
        observation = (
            f"Execution plan built with {len(plan)} steps: "
            + ", ".join(s['agent'] for s in plan) + "."
        )
        return plan, observation

    orchestrator_state, plan = react_run_loop(
        orchestrator_state,
        thought_text=(
            f"I have received inputs with image={'yes' if inputs.get('image') else 'no'} "
            f"and location='{inputs.get('location', 'not provided')}'. "
            f"I need to determine which agents to activate and in what order."
        ),
        action_name="build_execution_plan",
        action_fn=build_plan,
        action_input=inputs
    )

    orchestrator_state['config']['execution_plan'] = plan
    orchestrator_state = update_agent_memory(orchestrator_state, {
        'action': 'plan_created',
        'num_steps': len(plan)
    })

    return orchestrator_state, plan

def orchestrator_execute(orchestrator_state, plan, agents, context):
    """Orchestrator executes the plan by coordinating agents using a ReAct loop"""

    def run_all_steps(plan_and_context):
        plan, context, agents_copy = plan_and_context
        results = {}

        for step in sorted(plan, key=lambda x: x['priority']):
            agent_name = step['agent']
            agent_state = agents_copy.get(agent_name)

            if not agent_state:
                continue

            if agent_name == 'vision':
                agent_state, result = vision_agent_perceive(
                    agent_state,
                    context['image']
                )
                results['vision'] = result
                context['damage_info'] = result

            elif agent_name.startswith('cost_'):
                agent_state, result = cost_agent_estimate(
                    agent_state,
                    context['damage_info'],
                    context['location'],
                    context['currency']
                )
                results[agent_name] = result

            elif agent_name == 'shop_finder':
                agent_state, result = shop_finder_search(
                    agent_state,
                    context['location']
                )
                results['shops'] = result

            agents_copy[agent_name] = agent_state

        observation = (
            f"All {len(results)} agent tasks completed: "
            + ", ".join(results.keys()) + "."
        )
        return results, observation

    # Pack agents into the input tuple so the closure can access it
    orchestrator_state, results = react_run_loop(
        orchestrator_state,
        thought_text=(
            f"I have a plan with {len(plan)} steps. "
            f"I will execute each agent in priority order: vision first, "
            f"then cost estimators, then shop finder, "
            f"passing results downstream as context."
        ),
        action_name="execute_agent_plan",
        action_fn=lambda inp: run_all_steps(inp),
        action_input=(plan, context, agents)
    )

    orchestrator_state = update_agent_memory(orchestrator_state, {
        'action': 'execution_completed',
        'results_count': len(results)
    })

    return orchestrator_state, results

# --- ReAct Trace Formatter ---

def format_react_traces(orchestrator, agents):
    """Format all agents' ReAct traces into a readable markdown string"""
    ICONS = {'Thought': '💭', 'Action': '⚡', 'Observation': '👁️'}
    AGENT_LABELS = {
        'orchestrator': '🎯 Orchestrator',
        'vision':       '👁️ Vision Agent',
        'cost_primary': '💎 Cost Agent — OEM',
        'cost_alternative': '💰 Cost Agent — Aftermarket',
        'shop_finder':  '🛠️ Shop Finder Agent',
    }

    all_states = {'orchestrator': orchestrator, **agents}
    lines = []

    for agent_key, state in all_states.items():
        trace = state.get('react_trace', [])
        if not trace:
            continue
        label = AGENT_LABELS.get(agent_key, agent_key)
        lines.append(f"### {label}")
        for i, step in enumerate(trace, 1):
            icon = ICONS.get(step['step'], '•')
            ts = step['timestamp'][11:19]  # HH:MM:SS
            lines.append(f"**{icon} {step['step']}** <sub>{ts}</sub>")
            content = step['content']
            if step['step'] == 'Action' and step.get('input'):
                lines.append(f"> `{content}`")
            else:
                lines.append(f"> {content}")
            lines.append("")  # blank line between steps

    return "\n".join(lines) if lines else "_No trace available._"


# --- Main Analysis Function ---

def analyze_with_multi_agent_system(perplexity_key, openai_key, image, location):
    """
    Multi-agent system with orchestrator pattern:
    1. Orchestrator creates execution plan
    2. Vision agent analyzes image (autonomous decision on quality)
    3. Cost agents estimate in parallel (decide strategy based on severity)
    4. Shop finder agent searches (decides based on location)
    5. Orchestrator aggregates results
    """

    perplexity_key = perplexity_key or os.environ.get("PERPLEXITY_API_KEY")
    openai_key = openai_key or os.environ.get("OPENAI_API_KEY")

    if not perplexity_key:
        return "⚠️ Perplexity API Key Required", "N/A", "N/A", "", "_No trace — missing API key._"

    if not openai_key:
        return "⚠️ OpenAI API Key Required", "N/A", "N/A", "", "_No trace — missing API key._"

    if not image:
        return "Please upload an image", "N/A", "N/A", "", "_No trace — no image uploaded._"

    currency, currency_symbol = detect_currency_from_location(location)
    location = location or "European Union"

    try:
        orchestrator = create_orchestrator()

        agents = {
            'vision': create_vision_agent(openai_key),
            'cost_primary': create_cost_agent(perplexity_key, 'primary'),
            'cost_alternative': create_cost_agent(perplexity_key, 'alternative'),
            'shop_finder': create_shop_finder_agent(perplexity_key)
        }

        orchestrator, plan = orchestrator_plan(orchestrator, {
            'image': image,
            'location': location
        })

        context = {
            'image': image,
            'location': location,
            'currency': currency
        }

        orchestrator, results = orchestrator_execute(orchestrator, plan, agents, context)

        damage_info = results.get('vision', {})
        primary_cost = results.get('cost_primary', {})
        alternative_cost = results.get('cost_alternative', {})
        shops = results.get('shops', '')

        description_output = f"""<span style='background-color: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 4px; font-size: 0.75em;'>🤖 AI-Generated Analysis</span>

**Damage Analysis:**

{primary_cost.get('damage_summary', damage_info.get('description', 'N/A')[:500])}

**Affected Parts:** {', '.join(primary_cost.get('affected_parts', ['Not specified']))}

**Severity:** {damage_info.get('severity', 'Not determined')}"""

        low_primary = primary_cost.get('estimated_cost_low', 0)
        high_primary = primary_cost.get('estimated_cost_high', 0)

        primary_output = f"""<span style='background-color: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 4px; font-size: 0.75em;'>🤖 AI-Generated Estimate</span>

## 💎 {REPAIR_PHILOSOPHIES['primary']['name']}
### {currency_symbol}{low_primary:,.0f} - {currency_symbol}{high_primary:,.0f} {currency}

**Cost Breakdown:**
{primary_cost.get('cost_breakdown', 'N/A')}

**Warranty:** {primary_cost.get('warranty_info', 'N/A')}"""

        low_alt = alternative_cost.get('estimated_cost_low', 0)
        high_alt = alternative_cost.get('estimated_cost_high', 0)

        alternative_output = f"""<span style='background-color: #dbeafe; color: #1e40af; padding: 2px 8px; border-radius: 4px; font-size: 0.75em;'>🤖 AI-Generated Estimate</span>

## 💰 {REPAIR_PHILOSOPHIES['alternative']['name']}
### {currency_symbol}{low_alt:,.0f} - {currency_symbol}{high_alt:,.0f} {currency}

**Cost Breakdown:**
{alternative_cost.get('cost_breakdown', 'N/A')}

**Warranty:** {alternative_cost.get('warranty_info', 'N/A')}"""

        shops_output = f"""<span style='background-color: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 4px; font-size: 0.75em;'>🤖 AI-Generated Results</span>

### 🛠️ Local Repair Shops near {location}

{shops}"""

        return description_output, primary_output, alternative_output, shops_output, format_react_traces(orchestrator, agents)

    except Exception as e:
        return f"❌ Analysis Failed: {str(e)}", "N/A", "N/A", "", f"_Error during execution: {str(e)}_"


# --- Gradio Interface ---

def build_interface():
    with gr.Blocks(title="Multi-Agent Car Damage Evaluation") as demo:
        gr.Markdown(
            """
            <h1 style="text-align: center; font-size: 2.5em; color: #10b981;">🚗 Multi-Agent Car Damage Evaluation</h1>
            <p style="text-align: center; color: #6b7280; font-size: 1.1em;">
            Autonomous agents analyze damage, estimate costs, and find repair shops
            </p>
            <p style="text-align: center; color: #9ca3af; font-size: 0.9em;">
            Orchestrator coordinates: Vision Agent + Cost Agents (OEM/Aftermarket) + Shop Finder Agent
            </p>
            """
        )

        # AI Content Disclosure Banner
        gr.Markdown(
            """
            <div style="background-color: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; margin: 10px 0;">
                <p style="margin: 0; text-align: center; color: #92400e; font-size: 0.9em;">
                🤖 <strong>AI-Generated Content Notice:</strong> All analysis, cost estimates, and recommendations on this page are generated by artificial intelligence models (GPT-5.2 and Perplexity Sonar).
                Results should be verified by a professional before making any decisions.
                </p>
            </div>
            """
        )

        gr.Markdown("### 🔑 API Configuration")
        with gr.Row():
            perplexity_key_input = gr.Textbox(
                type="password",
                label="Perplexity API Key",
                placeholder="Enter your Perplexity API Key",
                scale=1
            )
            openai_key_input = gr.Textbox(
                type="password",
                label="OpenAI API Key",
                placeholder="Enter your OpenAI API Key",
                scale=1
            )

        gr.Markdown("---")
        gr.Markdown("### 📸 Damage Assessment")

        with gr.Row():
            image_input = gr.Image(type="pil", label="1. Upload Damage Photo", height=300)
            with gr.Column():
                location_input = gr.Textbox(
                    label="2. Enter Your Location",
                    placeholder="e.g., London, UK or Tokyo, Japan",
                    info="💡 Currency auto-detected"
                )
                gr.Markdown("**Supported:** USD, GBP, EUR, JPY, AUD, CAD, and more...")

        analyze_btn = gr.Button("🔍 Analyze with Multi-Agent System", variant="primary", size="lg")

        gr.Markdown("---")
        gr.Markdown("### 📊 Analysis Results <span style='background-color: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 4px; font-size: 0.7em; margin-left: 8px;'>🤖 AI-GENERATED</span>")

        description_output = gr.Markdown("Damage description will appear here...")

        gr.Markdown("### 💵 Comparative Cost Estimates <span style='background-color: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 4px; font-size: 0.7em; margin-left: 8px;'>🤖 AI-GENERATED</span>")

        with gr.Row():
            primary_cost_output = gr.Markdown("## Primary Quote\n*OEM Parts - Dealer*")
            alternative_cost_output = gr.Markdown("## Alternative Quote\n*Aftermarket - Independent*")

        shops_output = gr.Markdown("Shop results will appear here...")

        with gr.Accordion("🔍 ReAct Trace — Agent Reasoning Log", open=False):
            gr.Markdown(
                "<sub>Shows each agent's Thought → Action → Observation loop. Expand after running an analysis.</sub>"
            )
            react_trace_output = gr.Markdown("_Run an analysis to see the ReAct trace._")

        analyze_btn.click(
            fn=analyze_with_multi_agent_system,
            inputs=[perplexity_key_input, openai_key_input, image_input, location_input],
            outputs=[description_output, primary_cost_output, alternative_cost_output, shops_output, react_trace_output],
            show_progress=True
        )

        gr.Markdown(
            """
            ---
            <div style="background-color: #f3f4f6; border-radius: 8px; padding: 16px; margin-top: 20px;">
                <p style="text-align: center; color: #6b7280; font-size: 0.85em; margin-bottom: 12px;">
                <strong>Multi-Agent Architecture:</strong><br>
                🎯 <strong>Orchestrator Agent</strong> - Creates execution plan and coordinates agents<br>
                👁️ <strong>Vision Agent</strong> - Analyzes image and determines severity (autonomous decision-making)<br>
                💰 <strong>Cost Agents (2)</strong> - Generate estimates with different philosophies<br>
                🛠️ <strong>Shop Finder Agent</strong> - Searches based on location availability<br><br>
                Each agent maintains state, memory, and makes autonomous decisions!
                </p>
                <hr style="border: none; border-top: 1px solid #d1d5db; margin: 12px 0;">
                <p style="text-align: center; color: #9ca3af; font-size: 0.8em; margin: 0;">
                ⚠️ <strong>AI Content Disclaimer:</strong> All content on this page including damage analysis, cost estimates,
                and shop recommendations is generated by AI models (OpenAI GPT-5.2 and Perplexity Sonar-Pro).
                This information is provided for informational purposes only and should not be considered professional advice.
                Always consult with qualified auto repair professionals for accurate assessments and quotes.
                AI-generated estimates may not reflect actual repair costs in your area.
                </p>
            </div>
            """
        )

    return demo


# --- Launch ---

if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
