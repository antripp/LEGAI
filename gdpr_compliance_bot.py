import re
import json
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import plotly.graph_objects as go

class GDPRComplianceBot:
    def __init__(self):
        # Initialize with pre-trained models or rules
        self.model = None  # Placeholder for an ML model
        self.vectorizer = None
        self.ml_model = None
        self.flags = []
        self.passed_checks = []
        self.ml_flags = []

    def load_model(self):
        """
        Load or initialize the model for real-time code analysis.
        """
        # For now, we use basic rules; this can be extended with ML models
        self.model = {
            'data_collection': re.compile(r'\b(input|user_info|user_data)\b', re.IGNORECASE),
            'data_security': re.compile(r'\blog\b(?!\s*\()|\bwrite\b.*\bfile\b', re.IGNORECASE),
            'data_minimization': re.compile(r'\bminimize_data\b|\btrim\b', re.IGNORECASE),
            'proper_security': re.compile(r'\bhashlib\.sha256\b', re.IGNORECASE)
        }

        # Load vectorizer and ML model for advanced analysis
        try:
            self.vectorizer = joblib.load('vectorizer.joblib')
            self.ml_model = joblib.load('gdpr_ml_model.joblib')
        except FileNotFoundError:
            print("ML model or vectorizer not found. Make sure to train the model first.")

    def analyze_code(self, code_snippet):
        """
        Analyze the given code snippet for GDPR compliance issues.
        """
        if not self.model:
            self.load_model()

        issues = []

        # Rule-based analysis
        for area, pattern in self.model.items():
            matches = pattern.finditer(code_snippet)
            for match in matches:
                # Ignore correct use cases (e.g., `encrypt()` function or blank inputs)
                if area == 'data_security' and 'encrypt' in match.group().lower():
                    continue
                if 'input' in match.group().lower() and "''" in match.group():
                    continue

                issue = {
                    'type': area,
                    'line_number': code_snippet[:match.start()].count('\n') + 1,
                    'snippet': code_snippet[match.start():match.end()]
                }

                if area == 'proper_security':
                    issue['status'] = 'Compliance Check Passed'
                    self.passed_checks.append(issue)
                else:
                    issue['status'] = 'Compliance Issue Detected'
                    self.flags.append(issue)
                    issues.append(issue)

        # ML-based analysis
        if self.vectorizer and self.ml_model:
            code_lines = code_snippet.split('\n')
            for line_number, line in enumerate(code_lines, start=1):
                features = self.vectorizer.transform([line])
                prediction = self.ml_model.predict(features)
                if prediction == 1:  # Assuming 1 indicates a potential issue
                    # Assign a more specific issue type based on keywords in the line
                    issue_type = 'data_collection' if 'input' in line else 'data_security' if 'log' in line else 'data_minimization'
                    issue = {
                        'type': issue_type,
                        'line_number': line_number,
                        'snippet': line,
                        'status': 'Potential Issue Detected by ML'
                    }
                    self.ml_flags.append(issue)
                    issues.append(issue)

        return issues

    def reconcile_issues(self):
        """
        Reconcile ML-detected issues with rule-based detections and classify them accordingly.
        """
        reconciled_issues = []

        for ml_issue in self.ml_flags:
            matching_flag = next((f for f in self.flags if f['line_number'] == ml_issue['line_number'] and f['snippet'] == ml_issue['snippet']), None)
            if matching_flag:
                if matching_flag['status'] == 'Compliance Check Passed':
                    ml_issue['status'] = 'Passed but failed with ML'
                    ml_issue['highlight'] = 'orange'
                elif matching_flag['status'] == 'Compliance Issue Detected':
                    ml_issue['status'] = 'Failed and failed with ML'
                    ml_issue['highlight'] = 'deep red'
                reconciled_issues.append(ml_issue)
            else:
                if ml_issue['status'] == 'Potential Issue Detected by ML':
                    ml_issue['status'] = 'New issue identified by ML'
                    ml_issue['highlight'] = 'yellow'
                reconciled_issues.append(ml_issue)

        # Add issues that were only detected by rules
        for flag in self.flags:
            if not any(ml_issue['line_number'] == flag['line_number'] and ml_issue['snippet'] == flag['snippet'] for ml_issue in reconciled_issues):
                if flag['status'] == 'Compliance Check Passed':
                    flag['highlight'] = 'light green'
                elif flag['status'] == 'Compliance Issue Detected':
                    flag['highlight'] = 'light red'
                reconciled_issues.append(flag)

        self.flags = reconciled_issues

    def update_model(self, training_data, labels):
        """
        Update the model using RandomForestClassifier and GradientBoostingClassifier to improve accuracy based on new training data.
        """
        # Train a new Random Forest model with provided training data
        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(training_data)
        y = np.array(labels)

        # Use GridSearchCV for hyperparameter tuning
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestClassifier()
        grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=3, n_jobs=-1, verbose=1)
        grid_search_rf.fit(X, y)

        # Train a new Gradient Boosting model
        param_grid_gb = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10]
        }
        gb = GradientBoostingClassifier()
        grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=3, n_jobs=-1, verbose=1)
        grid_search_gb.fit(X, y)

        # Compare and choose the best model
        if grid_search_rf.best_score_ > grid_search_gb.best_score_:
            self.ml_model = grid_search_rf.best_estimator_
        else:
            self.ml_model = grid_search_gb.best_estimator_

        # Save the updated model and vectorizer
        joblib.dump(self.vectorizer, 'vectorizer.joblib')
        joblib.dump(self.ml_model, 'gdpr_ml_model.joblib')
        print("Model updated successfully with the best ML model.")

    def flag_issues(self, issues):
        """
        Flag the issues in real-time.
        """
        for issue in issues:
            print(f"{issue['status']}\nIssue Type: {issue['type']}\nCode Instance & Line: {issue['snippet']} (Line {issue['line_number']})\n")

    def generate_report(self, filename="compliance_report.json"):
        """
        Generate a report of all flagged issues in JSON format with structured sections.
        """
        failed_checks = []
        potential_issues = []
        passed_lookup_issues = []
        inconsistent_issues = []

        for issue in self.flags:
            if issue['status'] == 'Compliance Issue Detected':
                failed_checks.append(issue)
            elif issue['status'] == 'Potential Issue Detected by ML':
                potential_issues.append(issue)
            elif issue['status'] == 'Passed but failed with ML':
                passed_lookup_issues.append(issue)
            elif issue['status'] == 'Failed and failed with ML':
                inconsistent_issues.append(issue)

        report_data = {
            "Passed Checks": [
                {
                    'status': issue['status'],
                    'issue_type': issue['type'],
                    'code_instance': issue['snippet'],
                    'line_number': issue['line_number']
                }
                for issue in self.passed_checks
            ],
            "Failed Checks": [
                {
                    'status': issue['status'],
                    'issue_type': issue['type'],
                    'code_instance': issue['snippet'],
                    'line_number': issue['line_number']
                }
                for issue in failed_checks
            ],
            "New Issues Identified by ML": [
                {
                    'status': issue['status'],
                    'issue_type': issue['type'],
                    'code_instance': issue['snippet'],
                    'line_number': issue['line_number']
                }
                for issue in potential_issues
            ],
            "Passed but Failed with ML": [
                {
                    'status': issue['status'],
                    'issue_type': issue['type'],
                    'code_instance': issue['snippet'],
                    'line_number': issue['line_number']
                }
                for issue in passed_lookup_issues
            ],
            "Failed and Failed with ML": [
                {
                    'status': issue['status'],
                    'issue_type': issue['type'],
                    'code_instance': issue['snippet'],
                    'line_number': issue['line_number']
                }
                for issue in inconsistent_issues
            ]
        }

        with open(filename, 'w') as report_file:
            json.dump(report_data, report_file, indent=4)
        print(f"Compliance report generated: {filename}")

    def generate_visual_report(self, filename="compliance_report.html"):
        """
        Generate a visual report of the compliance analysis in a tabular format using Plotly.
        """
        # Prepare data for the table
        rows = []
        colors = []

        for issue in self.passed_checks:
            rows.append([issue['line_number'], issue['type'], issue['snippet'], 'Passed'])
            colors.append('#A1C349')  # Light green for passed rule checks

        for issue in self.flags:
            if issue['status'] == 'Compliance Issue Detected':
                rows.append([issue['line_number'], issue['type'], issue['snippet'], 'Failed'])
                colors.append('#EC5247')  # Light red for failed rule checks
            elif issue['status'] == 'Passed but failed with ML':
                rows.append([issue['line_number'], issue['type'], issue['snippet'], 'Passed but failed with ML'])
                colors.append('#FFA500')  # Orange for needs lookup
            elif issue['status'] == 'Failed and failed with ML':
                rows.append([issue['line_number'], issue['type'], issue['snippet'], 'Failed and failed with ML'])
                colors.append('#CC3227')  # Deep red for completely failed
            elif issue['status'] == 'New issue identified by ML':
                rows.append([issue['line_number'], issue['type'], issue['snippet'], 'New issue identified by ML'])
                colors.append('#FF9F1C')  # Yellow for new issues identified by ML
            elif issue['status'] == 'Failed but passed with ML':
                rows.append([issue['line_number'], issue['type'], issue['snippet'], 'Failed but passed with ML'])
                colors.append('#5BC0EB')  # Light blue for needs lookup
            elif issue['status'] == 'Passed and passed with ML':
                rows.append([issue['line_number'], issue['type'], issue['snippet'], 'Passed and passed with ML'])
                colors.append('#87A330')  # Dark green for completely passed

        # Create Plotly table
        header_color = '#404040'
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Line Number', 'Issue Type', 'Code Instance', 'Status'],
                fill_color=header_color,
                align='center',
                font=dict(color='white', size=14)
            ),
            cells=dict(
                values=[list(x) for x in zip(*rows)],
                fill_color=[colors],
                align='center',
                font=dict(size=12)
            )
        )])

        # Save the table to an HTML file
        fig.write_html(filename)
        print(f"Visual compliance report generated: {filename}")

    def run(self, code_snippet):
        """
        Run the bot to analyze code, flag issues, generate a report, and update the model.
        """
        # Step 1: Real-time code analysis
        issues = self.analyze_code(code_snippet)

        # Step 2: Reconcile ML and rule-based issues
        self.reconcile_issues()

        # Step 3: Flag the issues in real-time
        self.flag_issues(issues)

        # Step 4: Generate a report
        self.generate_report()

        # Step 5: Generate a visual report
        self.generate_visual_report()

    # Added the missing method to prevent AttributeError
    def iterative_model_update(self, code_snippet, max_iterations=10):
        """
        Continuously update the model until the number of ML-detected issues is reduced
        to match the number of failed compliance issues or until the maximum number of iterations is reached.
        After each iteration, re-analyze the code without generating intermediate reports.
        """
        iteration = 0
        previous_ml_issues = len([issue for issue in self.flags if issue['status'] == 'Potential Issue Detected by ML'])
        failed_issues = len([issue for issue in self.flags if issue['status'] == 'Compliance Issue Detected'])

        while iteration < max_iterations and previous_ml_issues > failed_issues:
            print(f"Iteration {iteration + 1}: Updating model...")
            self.update_model_with_user_code(code_snippet)
            iteration += 1

            # Re-analyze the code after updating the model
            self.flags = []
            self.passed_checks = []
            self.ml_flags = []
            issues = self.analyze_code(code_snippet)

            previous_ml_issues = len([issue for issue in self.flags if issue['status'] == 'Potential Issue Detected by ML'])
            failed_issues = len([issue for issue in self.flags if issue['status'] == 'Compliance Issue Detected'])

            print(f"Number of ML-detected issues: {previous_ml_issues}, Failed issues: {failed_issues}")
            if previous_ml_issues <= failed_issues:
                print("ML-detected issues have been reduced to match the number of failed issues.")
                break

        if iteration == max_iterations:
            print("Maximum number of iterations reached. Model update completed.")

        # Generate final report after iterations
        self.generate_report("compliance_report_final.json")
        self.generate_visual_report("compliance_report_final.html")