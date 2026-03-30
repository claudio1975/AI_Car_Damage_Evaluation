import streamlit as st
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
    "DE": "EUR", "Germany": "EUR", "Berlin": "EUR", "Munich": "EUR",
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
    return {
        'type': agent_type,
        'memory': [],
        'confidence': 1.0,
        'last_action': None,
        'timestamp': datetime.now().isoformat(),
        'config': config or {},
        'performance_history': []
    }

def update_agent_memory(agent_state, event):
    agent_state['memory'].append({
        'timestamp': datetime.now().isoformat(),
        'event': event
    })
    if len(agent_state['memory']) > 10:
        agent_state['memory'] = agent_state['memory'][-10:]
    return agent_state

def agent_decide(agent_state, context):
    decision = {
        'action': None,
        'confidence': agent_state['confidence'],
        'reasoning': []
    }
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

# --- Helper Functions ---

def detect_currency_from_location(location):
    if not location:
        return "USD", "$"
    location_clean = location.strip()
    for key, currency in CURRENCY_MAP.items():
        if key.lower() in location_clean.lower():
            return currency, CURRENCY_SYMBOLS.get(currency, currency)
    return "USD", "$"

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# --- Vision Agent ---

def create_vision_agent(api_key):
    return create_agent_state('vision', {
        'api_key': api_key,
        'confidence_threshold': 0.7
    })

def vision_agent_perceive(agent_state, image):
    decision = agent_decide(agent_state, {'has_image': image is not None})
    if decision['action'] != 'analyze_image':
        return agent_state, None

    headers = {
        "Authorization": f"Bearer {agent_state['config']['api_key']}",
        "Content-Type": "application/json"
    }
    img_base64 = image_to_base64(image)
    payload = {
        "model": OPENAI_VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert auto damage evaluator. Analyze images with precision, detail and assess damage severity."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyze this car damage image. Provide:
1. Overall description of the damage
2. Specific parts that are damaged (be precise with part names)
3. Severity of damage (minor, moderate, severe)
4. Type of damage (collision, scratch, dent, etc.)
5. Any additional observations

Be specific and technical in your assessment."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    }
                ]
            }
        ],
        "max_completion_tokens": 1000
    }

    response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    analysis = result['choices'][0]['message']['content']

    severity = 'moderate'
    if 'severe' in analysis.lower() or 'major' in analysis.lower():
        severity = 'severe'
    elif 'minor' in analysis.lower() or 'light' in analysis.lower():
        severity = 'minor'

    agent_state = update_agent_memory(agent_state, {
        'action': 'image_analyzed',
        'severity_detected': severity,
        'confidence': 0.9
    })
    return agent_state, {
        'description': analysis,
        'severity': severity,
        'confidence': 0.9
    }

# --- Cost Estimation Agent ---

def create_cost_agent(api_key, philosophy):
    return create_agent_state('cost_estimator', {
        'api_key': api_key,
        'philosophy': philosophy,
        'price_adjustment_factor': 1.0
    })

def cost_agent_estimate(agent_state, damage_info, location, currency):
    philosophy_key = agent_state['config']['philosophy']
    philosophy = REPAIR_PHILOSOPHIES[philosophy_key]
    currency_symbol = CURRENCY_SYMBOLS.get(currency, currency)

    headers = {
        "Authorization": f"Bearer {agent_state['config']['api_key']}",
        "Content-Type": "application/json"
    }

    prompt = f"""Based on this car damage, provide repair cost estimate for {location} using {philosophy['description']}:

Damage: {damage_info['description']}
Severity: {damage_info.get('severity', 'moderate')}

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

    payload = {
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

    response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    response_text = result['choices'][0]['message']['content']

    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        cost_data = json.loads(json_match.group())
    else:
        cost_data = json.loads(response_text)

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
        cost_data['cost_breakdown'] = breakdown_text if len(breakdown_text) > 20 else f"Estimate includes {philosophy['parts_type']} and {philosophy['labor_source']} labor rates for {location}."

    agent_state = update_agent_memory(agent_state, {
        'action': 'estimate_generated',
        'philosophy': philosophy_key,
        'cost_range': f"{cost_data.get('estimated_cost_low')}-{cost_data.get('estimated_cost_high')}"
    })
    return agent_state, cost_data

# --- Shop Finder Agent ---

def create_shop_finder_agent(api_key):
    return create_agent_state('shop_finder', {
        'api_key': api_key,
        'search_radius': 'default'
    })

def shop_finder_search(agent_state, location):
    decision = agent_decide(agent_state, {'location': location})
    if decision['action'] == 'skip':
        return agent_state, "Enter a location to find nearby repair shops."

    headers = {
        "Authorization": f"Bearer {agent_state['config']['api_key']}",
        "Content-Type": "application/json"
    }
    payload = {
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
        "temperature": 0.3
    }

    response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    shops_text = result['choices'][0]['message']['content']

    agent_state = update_agent_memory(agent_state, {
        'action': 'shops_found',
        'location': location
    })
    return agent_state, shops_text

# --- Orchestrator Agent ---

def create_orchestrator():
    return create_agent_state('orchestrator', {
        'execution_plan': [],
        'agent_registry': {}
    })

def orchestrator_plan(orchestrator_state, inputs):
    plan = []
    if inputs.get('image'):
        plan.append({'agent': 'vision', 'priority': 1, 'reason': 'Image analysis required'})
    if inputs.get('location'):
        plan.append({'agent': 'cost_primary', 'priority': 2, 'reason': 'Primary quote needed'})
        plan.append({'agent': 'cost_alternative', 'priority': 2, 'reason': 'Alternative quote for comparison'})
        plan.append({'agent': 'shop_finder', 'priority': 3, 'reason': 'Find local repair shops'})
    orchestrator_state['config']['execution_plan'] = plan
    orchestrator_state = update_agent_memory(orchestrator_state, {
        'action': 'plan_created',
        'num_steps': len(plan)
    })
    return orchestrator_state, plan

def orchestrator_execute(orchestrator_state, plan, agents, context):
    results = {}
    for step in sorted(plan, key=lambda x: x['priority']):
        agent_name = step['agent']
        agent_state = agents.get(agent_name)
        if not agent_state:
            continue
        if agent_name == 'vision':
            agent_state, result = vision_agent_perceive(agent_state, context['image'])
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
            agent_state, result = shop_finder_search(agent_state, context['location'])
            results['shops'] = result
        agents[agent_name] = agent_state

    orchestrator_state = update_agent_memory(orchestrator_state, {
        'action': 'execution_completed',
        'results_count': len(results)
    })
    return orchestrator_state, results

# --- Main Analysis Function ---

def analyze_with_multi_agent_system(perplexity_key, openai_key, image, location):
    perplexity_key = perplexity_key or os.environ.get("PERPLEXITY_API_KEY")
    openai_key = openai_key or os.environ.get("OPENAI_API_KEY")

    if not perplexity_key:
        return None, "⚠️ Perplexity API Key Required", None, None, None
    if not openai_key:
        return None, "⚠️ OpenAI API Key Required", None, None, None
    if not image:
        return None, "Please upload an image", None, None, None

    currency, currency_symbol = detect_currency_from_location(location)
    location = location or "United States"

    orchestrator = create_orchestrator()
    agents = {
        'vision': create_vision_agent(openai_key),
        'cost_primary': create_cost_agent(perplexity_key, 'primary'),
        'cost_alternative': create_cost_agent(perplexity_key, 'alternative'),
        'shop_finder': create_shop_finder_agent(perplexity_key)
    }
    orchestrator, plan = orchestrator_plan(orchestrator, {'image': image, 'location': location})
    context = {'image': image, 'location': location, 'currency': currency}
    orchestrator, results = orchestrator_execute(orchestrator, plan, agents, context)

    damage_info = results.get('vision', {})
    primary_cost = results.get('cost_primary', {})
    alternative_cost = results.get('cost_alternative', {})
    shops = results.get('shops', '')

    return damage_info, primary_cost, alternative_cost, shops, currency_symbol, currency, location

# --- Streamlit UI ---

st.set_page_config(
    page_title="Multi-Agent Car Damage Evaluation",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<h1 style="text-align:center; color:#10b981;">🚗 Multi-Agent Car Damage Evaluation</h1>
<p style="text-align:center; color:#6b7280; font-size:1.1em;">
    Autonomous agents analyze damage, estimate costs, and find repair shops
</p>
<p style="text-align:center; color:#9ca3af; font-size:0.9em;">
    Orchestrator coordinates: Vision Agent + Cost Agents (OEM/Aftermarket) + Shop Finder Agent
</p>
""", unsafe_allow_html=True)

st.info(
    "🤖 **AI-Generated Content Notice:** All analysis, cost estimates, and recommendations are generated by "
    "AI models (GPT-5.2 and Perplexity Sonar). Results should be verified by a professional before making any decisions."
)

# --- API Configuration ---
st.markdown("### 🔑 API Configuration")
col1, col2 = st.columns(2)
with col1:
    perplexity_key_input = st.text_input(
        "Perplexity API Key",
        type="password",
        placeholder="Enter your Perplexity API Key"
    )
with col2:
    openai_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your OpenAI API Key"
    )

st.markdown("---")
st.markdown("### 📸 Damage Assessment")

col_img, col_loc = st.columns([1, 1])
with col_img:
    uploaded_file = st.file_uploader(
        "1. Upload Damage Photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear photo of the car damage"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    else:
        image = None

with col_loc:
    location_input = st.text_input(
        "2. Enter Your Location",
        placeholder="e.g., London, UK or Tokyo, Japan",
        help="💡 Currency is auto-detected from your location"
    )
    st.caption("**Supported currencies:** USD, GBP, EUR, JPY, AUD, CAD, and more...")
    if location_input:
        currency, symbol = detect_currency_from_location(location_input)
        st.success(f"Detected currency: **{symbol} {currency}**")

analyze_btn = st.button(
    "🔍 Analyze with Multi-Agent System",
    type="primary",
    use_container_width=True
)

# --- Results ---
if analyze_btn:
    if not image:
        st.error("Please upload a damage photo before analyzing.")
    elif not perplexity_key_input and not os.environ.get("PERPLEXITY_API_KEY"):
        st.error("⚠️ Perplexity API Key is required.")
    elif not openai_key_input and not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ OpenAI API Key is required.")
    else:
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")

        with st.spinner("🤖 Multi-agent system is analyzing your damage... Please wait."):
            try:
                result = analyze_with_multi_agent_system(
                    perplexity_key_input,
                    openai_key_input,
                    image,
                    location_input
                )
                # Unpack
                if len(result) == 5:
                    damage_info, primary_cost, alternative_cost, shops, err = result
                    if err and not isinstance(primary_cost, dict):
                        st.error(err)
                        st.stop()
                else:
                    damage_info, primary_cost, alternative_cost, shops, currency_symbol, currency, location_used = result

                # Re-detect for display
                currency, currency_symbol = detect_currency_from_location(location_input)
                location_used = location_input or "United States"

                # --- Damage Description ---
                st.markdown("#### 🔎 Damage Analysis")
                st.markdown(
                    '<span style="background:#e0e7ff;color:#3730a3;padding:2px 8px;border-radius:4px;font-size:0.75em;">🤖 AI-Generated Analysis</span>',
                    unsafe_allow_html=True
                )
                if isinstance(damage_info, dict):
                    description = primary_cost.get('damage_summary', damage_info.get('description', 'N/A'))
                    affected_parts = primary_cost.get('affected_parts', ['Not specified'])
                    severity = damage_info.get('severity', 'Not determined')

                    with st.container():
                        st.markdown(f"**Damage Analysis:**\n\n{description}")
                        st.markdown(f"**Affected Parts:** {', '.join(affected_parts)}")

                        severity_color = {"minor": "green", "moderate": "orange", "severe": "red"}.get(
                            str(severity).lower(), "gray"
                        )
                        st.markdown(f"**Severity:** :{severity_color}[{severity.upper()}]")
                elif isinstance(damage_info, str):
                    st.error(damage_info)
                    st.stop()

                st.markdown("---")

                # --- Cost Estimates ---
                st.markdown("### 💵 Comparative Cost Estimates")
                st.markdown(
                    '<span style="background:#e0e7ff;color:#3730a3;padding:2px 8px;border-radius:4px;font-size:0.75em;">🤖 AI-Generated</span>',
                    unsafe_allow_html=True
                )

                col_primary, col_alt = st.columns(2)

                with col_primary:
                    st.markdown(
                        '<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;font-size:0.75em;">🤖 AI-Generated Estimate</span>',
                        unsafe_allow_html=True
                    )
                    if isinstance(primary_cost, dict):
                        low_p = primary_cost.get('estimated_cost_low', 0)
                        high_p = primary_cost.get('estimated_cost_high', 0)
                        st.markdown(f"## 💎 {REPAIR_PHILOSOPHIES['primary']['name']}")
                        st.markdown(f"### {currency_symbol}{low_p:,.0f} – {currency_symbol}{high_p:,.0f} {currency}")
                        with st.expander("Cost Breakdown", expanded=True):
                            st.markdown(primary_cost.get('cost_breakdown', 'N/A'))
                        st.markdown(f"**Warranty:** {primary_cost.get('warranty_info', 'N/A')}")
                    else:
                        st.info("Primary cost estimate not available.")

                with col_alt:
                    st.markdown(
                        '<span style="background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:4px;font-size:0.75em;">🤖 AI-Generated Estimate</span>',
                        unsafe_allow_html=True
                    )
                    if isinstance(alternative_cost, dict):
                        low_a = alternative_cost.get('estimated_cost_low', 0)
                        high_a = alternative_cost.get('estimated_cost_high', 0)
                        st.markdown(f"## 💰 {REPAIR_PHILOSOPHIES['alternative']['name']}")
                        st.markdown(f"### {currency_symbol}{low_a:,.0f} – {currency_symbol}{high_a:,.0f} {currency}")
                        with st.expander("Cost Breakdown", expanded=True):
                            st.markdown(alternative_cost.get('cost_breakdown', 'N/A'))
                        st.markdown(f"**Warranty:** {alternative_cost.get('warranty_info', 'N/A')}")
                    else:
                        st.info("Alternative cost estimate not available.")

                st.markdown("---")

                # --- Shop Finder ---
                if shops:
                    st.markdown(
                        '<span style="background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px;font-size:0.75em;">🤖 AI-Generated Results</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f"### 🛠️ Local Repair Shops near {location_used}")
                    st.markdown(shops)

            except Exception as e:
                st.error(f"❌ Analysis Failed: {str(e)}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="background:#f3f4f6;border-radius:8px;padding:16px;margin-top:20px;">
    <p style="text-align:center;color:#6b7280;font-size:0.85em;margin-bottom:12px;">
        <strong>Multi-Agent Architecture:</strong><br>
        🎯 <strong>Orchestrator Agent</strong> – Creates execution plan and coordinates agents<br>
        👁️ <strong>Vision Agent</strong> – Analyzes image and determines severity (autonomous decision-making)<br>
        💰 <strong>Cost Agents (2)</strong> – Generate estimates with different philosophies<br>
        🛠️ <strong>Shop Finder Agent</strong> – Searches based on location availability<br><br>
        Each agent maintains state, memory, and makes autonomous decisions!
    </p>
    <hr style="border:none;border-top:1px solid #d1d5db;margin:12px 0;">
    <p style="text-align:center;color:#9ca3af;font-size:0.8em;margin:0;">
        ⚠️ <strong>AI Content Disclaimer:</strong> All content including damage analysis, cost estimates,
        and shop recommendations is generated by AI models (OpenAI GPT-5.2 and Perplexity Sonar-Pro).
        This information is for informational purposes only and should not be considered professional advice.
        Always consult qualified auto repair professionals for accurate assessments and quotes.
    </p>
</div>
""", unsafe_allow_html=True)
