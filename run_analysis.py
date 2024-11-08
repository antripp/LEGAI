from gdpr_compliance_bot import GDPRComplianceBot
#from frontend_compliance_bot import FrontendComplianceBot
# Sample e-commerce code snippet
with open("golf_test.py", "r") as f:
    ecommerce_code = f.read()

# Create an instance of GDPRComplianceBot and run the analysis
bot = GDPRComplianceBot()
#bot = FrontendComplianceBot()
bot.run(ecommerce_code)
