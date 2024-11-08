import json
import random
from faker import Faker
from gdpr_compliance_bot import GDPRComplianceBot
import string

# Script to create training samples for GDPR Compliance Bot
def generate_training_samples(num_samples=1000000):
    fake = Faker()
    # Base examples to create variations
    base_training_data = [
        "user_info = input('Enter your information: ')",  # data_collection issue
        "encrypt(user_info)",  # Proper data security (pass)
        "log(user_info)",  # data_security issue
        "trim_data(user_info)",  # data_minimization issue
        "secure_payment(payment_details)",  # Proper data security (pass)
        "payment_details = input('Enter payment details: ')",  # data_collection issue
        "write_to_file(user_data)",  # data_security issue
        "minimize_data(user_info)",  # data_minimization issue
        "store_user_data(user_info)",  # data_collection issue
        "encrypt(payment_details)",  # Proper data security (pass)
        "log_payment_info(payment_info)",  # data_security issue
        "user_name = input('Enter your name: ')",  # data_collection issue
        "store_sensitive_data(user_data)",  # data_collection issue
        "encrypt_sensitive_data(sensitive_info)",  # Proper data security (pass)
        "write_sensitive_log(sensitive_data)",  # data_security issue
        "delete_unused_data(user_data)",  # data_minimization issue
        "collect_user_feedback(feedback)",  # data_collection issue
        "log_user_activity(activity)",  # data_security issue
        "hash_password(user_password)",  # Proper data security (pass)
        "retain_user_data(user_data, retention_period)",  # data_minimization issue
        "sanitize_input(user_input)",  # Proper data minimization (pass)
        "store_payment_details(payment_details)",  # data_collection issue
        "log_sensitive_action(action_details)",  # data_security issue
        "encrypt_password(user_password)",  # Proper data security (pass)
        "remove_old_logs(log_data)",  # data_minimization issue
        "store_contact_info(contact_info)",  # data_collection issue
        "log_transaction(transaction_details)",  # data_security issue
        "secure_store(user_password)",  # Proper data security (pass)
        "retain_sensitive_logs(logs, retention_policy)",  # data_minimization issue
        "input_user_credentials()",  # data_collection issue
        "store_user_credentials(credentials)",  # data_collection issue
        "encrypt_user_credentials(credentials)",  # Proper data security (pass)
        "log_credentials(credentials)",  # data_security issue
        "minimize_stored_user_data(user_data)",  # data_minimization issue
        "user_age = input('Enter your age: ')",  # data_collection issue
        "user_email = input('Enter your email address: ')",  # data_collection issue
        "password_hash = hashlib.sha256(password.encode()).hexdigest()",  # Proper data security (pass)
        "with open('log.txt', 'w') as log_file: log_file.write(log_data)",  # data_security issue
        "user_feedback = request.get_json()",  # data_collection issue
        "if sensitive_info: log_sensitive_data(sensitive_info)",  # data_security issue
        "delete_personal_info(user_info)",  # Proper data minimization (pass)
        "store_sensitive_information(user_social_security_number)",  # data_collection issue
        "hashed_token = hashlib.sha512(token.encode()).hexdigest()",  # Proper data security (pass)
        "log_user_login_attempt(user_login_info)",  # data_security issue
        "anonymize_user_data(user_data)",  # Proper data minimization (pass)
        "store_user_location(location_data)",  # data_collection issue
        "write_log(log_entry)",  # data_security issue
    ]

    base_labels = [
        1,  # data_collection issue
        0,  # Proper data security (pass)
        1,  # data_security issue
        1,  # data_minimization issue
        0,  # Proper data security (pass)
        1,  # data_collection issue
        1,  # data_security issue
        1,  # data_minimization issue
        1,  # data_collection issue
        0,  # Proper data security (pass)
        1,  # data_security issue
        1,  # data_collection issue
        1,  # data_collection issue
        0,  # Proper data security (pass)
        1,  # data_security issue
        1,  # data_minimization issue
        1,  # data_collection issue
        1,  # data_security issue
        0,  # Proper data security (pass)
        1,  # data_minimization issue
        0,  # Proper data minimization (pass)
        1,  # data_collection issue
        1,  # data_security issue
        0,  # Proper data security (pass)
        1,  # data_minimization issue
        1,  # data_collection issue
        1,  # data_security issue
        0,  # Proper data security (pass)
        1,  # data_minimization issue
        1,  # data_collection issue
        1,  # data_collection issue
        0,  # Proper data security (pass)
        1,  # data_security issue
        1,  # data_minimization issue
        1,  # data_collection issue
        1,  # data_collection issue
        0,  # Proper data security (pass)
        1,  # data_security issue
        1,  # data_security issue
        1,  # data_collection issue
        1,  # data_security issue
        0,  # Proper data minimization (pass)
        1,  # data_collection issue
        0,  # Proper data security (pass)
        1,  # data_security issue
        0,  # Proper data minimization (pass)
        1,  # data_collection issue
        0,  # Proper data security (pass)
        1,  # data_security issue
    ]

    # Generate random samples from base examples with variations
    training_data = []
    labels = []
    for _ in range(num_samples):
        idx = random.randint(0, len(base_training_data) - 1)
        sample = base_training_data[idx]

        # Add variations to the sample using Faker and random modifications
        if "user_info" in sample or "payment_details" in sample or "user_data" in sample or "credentials" in sample or "contact_info" in sample:
            sample = sample.replace("user_info", fake.first_name())
            sample = sample.replace("payment_details", fake.credit_card_number())
            sample = sample.replace("user_data", fake.email())
            sample = sample.replace("credentials", fake.password())
            sample = sample.replace("contact_info", fake.phone_number())
        elif "activity" in sample or "feedback" in sample or "sensitive_data" in sample or "action_details" in sample or "transaction_details" in sample:
            sample = sample.replace("activity", fake.sentence())
            sample = sample.replace("feedback", fake.paragraph())
            sample = sample.replace("sensitive_data", fake.ssn())
            sample = sample.replace("action_details", fake.sentence())
            sample = sample.replace("transaction_details", fake.text())

        # Randomly modify the structure to add variety
        if random.random() > 0.5:
            sample = sample.replace(" ", "  ")  # Add extra spaces for variation
        if random.random() > 0.7:
            sample = sample.replace("=", " = ")  # Add spaces around equal signs
        if random.random() > 0.8:
            sample = sample.replace("(", " ( ").replace(")", " ) ")  # Add spaces around parentheses
        if random.random() > 0.9:
            sample = f"# Random comment\n{sample}"  # Add random comment lines
        if random.random() > 0.85:
            sample = sample.upper()  # Convert to uppercase to simulate inconsistent coding styles
        if random.random() > 0.95:
            sample = sample + "\n# Additional debug log"

        training_data.append(sample)
        labels.append(base_labels[idx])

    # Save the training samples and labels as a JSON file
    training_samples = {
        "data": training_data,
        "labels": labels
    }

    with open('training_samples.json', 'w') as outfile:
        json.dump(training_samples, outfile, indent=4)

    print("Training samples saved to 'training_samples.json'")

    return training_data, labels

# Function to update the GDPR Compliance Bot model
def update_gdpr_compliance_bot(training_data, labels):
    # Create an instance of the GDPRComplianceBot
    bot = GDPRComplianceBot()

    # Update the model with the training data and labels
    bot.update_model(training_data, labels)
    print("Model updated successfully with new training data.")

if __name__ == "__main__":
    # Generate training samples
    training_data, labels = generate_training_samples()

    # Update the GDPR Compliance Bot model
    update_gdpr_compliance_bot(training_data, labels)