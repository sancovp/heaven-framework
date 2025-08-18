#!/usr/bin/env python3
"""
HEAVEN Dynamic Tool Creation Example
Shows how to create tools dynamically using make_heaven_tool_from_docstring
"""

import asyncio
import os
os.environ['HEAVEN_DATA_DIR'] = '/tmp/heaven_data'

from heaven_base import (
    BaseHeavenAgent,
    HeavenAgentConfig,
    UnifiedChat,
    ProviderEnum,
    make_heaven_tool_from_docstring
)
from heaven_base.memory.history import History

# Define some utility functions that we'll turn into tools
def calculate_tax(amount: float, tax_rate: float) -> dict:
    """
    Calculate tax and total amount for a purchase.
    
    This tool calculates the tax amount and total cost including tax.
    
    Args:
        amount (float): The base amount before tax
        tax_rate (float): The tax rate as a decimal (e.g., 0.08 for 8%)
    
    Returns:
        dict: A dictionary containing the base amount, tax amount, and total
    """
    tax_amount = amount * tax_rate
    total = amount + tax_amount
    
    return {
        "base_amount": amount,
        "tax_rate": tax_rate,
        "tax_amount": round(tax_amount, 2),
        "total_amount": round(total, 2),
        "formatted_summary": f"Base: ${amount:.2f}, Tax: ${tax_amount:.2f}, Total: ${total:.2f}"
    }

def generate_password(length: int = 12, include_symbols: bool = True) -> dict:
    """
    Generate a secure random password.
    
    This tool creates a cryptographically secure password with customizable options.
    
    Args:
        length (int): Length of the password (default: 12, minimum: 4)
        include_symbols (bool): Whether to include special symbols (default: True)
    
    Returns:
        dict: A dictionary containing the generated password and its strength info
    """
    import random
    import string
    
    # Validate length
    if length < 4:
        length = 4
    
    # Build character set
    chars = string.ascii_letters + string.digits
    if include_symbols:
        chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    # Generate password
    password = ''.join(random.choice(chars) for _ in range(length))
    
    # Calculate strength
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_symbol = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    strength_score = sum([has_upper, has_lower, has_digit, has_symbol])
    strength_levels = ["Weak", "Fair", "Good", "Strong"]
    strength = strength_levels[min(strength_score - 1, 3)] if strength_score > 0 else "Very Weak"
    
    return {
        "password": password,
        "length": length,
        "strength": strength,
        "includes_symbols": include_symbols,
        "strength_details": {
            "has_uppercase": has_upper,
            "has_lowercase": has_lower,
            "has_digits": has_digit,
            "has_symbols": has_symbol
        }
    }

def convert_temperature(temperature: float, from_unit: str, to_unit: str) -> dict:
    """
    Convert temperature between Celsius, Fahrenheit, and Kelvin.
    
    This tool performs temperature conversions between different units.
    
    Args:
        temperature (float): The temperature value to convert
        from_unit (str): Source unit ('C', 'F', or 'K')
        to_unit (str): Target unit ('C', 'F', or 'K')
    
    Returns:
        dict: A dictionary containing the conversion result and details
    """
    # Normalize units
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    
    # Convert to Celsius first
    if from_unit == 'F':
        celsius = (temperature - 32) * 5/9
    elif from_unit == 'K':
        celsius = temperature - 273.15
    else:  # from_unit == 'C'
        celsius = temperature
    
    # Convert from Celsius to target
    if to_unit == 'F':
        result = celsius * 9/5 + 32
    elif to_unit == 'K':
        result = celsius + 273.15
    else:  # to_unit == 'C'
        result = celsius
    
    return {
        "original_temperature": temperature,
        "original_unit": from_unit,
        "converted_temperature": round(result, 2),
        "target_unit": to_unit,
        "formatted_result": f"{temperature}°{from_unit} = {result:.2f}°{to_unit}"
    }

async def main():
    print("=== HEAVEN Dynamic Tool Creation Example ===\n")
    
    # Create tools dynamically from our functions using docstrings
    print("Creating tools from function docstrings...")
    
    TaxCalculatorTool = make_heaven_tool_from_docstring(calculate_tax)
    PasswordGeneratorTool = make_heaven_tool_from_docstring(generate_password)
    TemperatureConverterTool = make_heaven_tool_from_docstring(convert_temperature)
    
    print("✓ Created TaxCalculatorTool")
    print("✓ Created PasswordGeneratorTool") 
    print("✓ Created TemperatureConverterTool")
    
    # Create agent configuration with our dynamically created tools
    config = HeavenAgentConfig(
        name="UtilityAgent",
        system_prompt="""You are a helpful utility agent with access to calculation and conversion tools.

You have access to:
- Tax calculator for purchase calculations
- Password generator for secure passwords
- Temperature converter for unit conversions

Always use the appropriate tool when users ask for calculations or conversions.""",
        tools=[TaxCalculatorTool, PasswordGeneratorTool, TemperatureConverterTool],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.3
    )
    
    # Initialize components
    history = History(messages=[])
    
    # Create the agent
    agent = BaseHeavenAgent(config, UnifiedChat, history=history)
    
    # Test the dynamically created tools
    test_prompts = [
        "Calculate the tax and total cost for a $150 purchase with 8.5% sales tax",
        "Generate a 16-character password with symbols",
        "Convert 25 degrees Celsius to Fahrenheit"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n=== Test {i}: Dynamic Tool Usage ===")
        print(f"User: {prompt}\n")
        
        result = await agent.run(prompt=prompt)
        
        # Display the response
        if isinstance(result, dict) and "history" in result:
            for msg in result["history"].messages:
                if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                    print(f"Assistant: {msg.content}")
                    break
    
    # Show the capabilities of the dynamic tools
    print(f"\n=== Dynamic Tool Information ===")
    print(f"✓ All tools were created from Python function docstrings")
    print(f"✓ Tools automatically parse function signatures for arguments")
    print(f"✓ Docstrings become tool descriptions for the agent")
    print(f"✓ Type hints are used for parameter validation")
    
    # Show the final history ID
    history_id = result.get("history_id")
    print(f"\nHistory ID: {history_id}")

if __name__ == "__main__":
    asyncio.run(main())