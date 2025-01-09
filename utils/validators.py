import re
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone):
    # Basic phone validation - can be adjusted based on your needs
    pattern = r'^\+?1?\d{9,15}$'
    return bool(re.match(pattern, phone))

def validate_tech_stack(tech_stack):
    techs = [tech.strip() for tech in tech_stack.split(',') if tech.strip()]
    return len(techs) > 0
