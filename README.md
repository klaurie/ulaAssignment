# ulaAssignment
ULA Placement Optimization Program
Overview
This optimization program is designed to allocate Undergraduate Learning Assistants (ULAs) to studios for a given set of constraints and preferences, considering several factors such as ULA availability, instructor preferences, studio requirements, and demographic considerations (e.g., major, race, gender). The program uses Gurobi optimization to determine the best assignments based on multiple objectives while ensuring fairness and adherence to constraints.

Features
Studio Assignment: Assign ULAs to studios ensuring they meet the studio's required number of ULAs and matching language preferences.
Instructor Preferences: Ensure ULAs are assigned to instructors they prefer, based on ULA preferences.
Student Availability: Ensure ULAs are assigned to studios where they are available.
Demographic Constraints: Enforce fairness in the assignment process by considering ULA demographics (e.g., gender, race, major, school, and honors college).
E-campus Preferences: Manage the assignment of e-campus and on-campus students to corresponding studio types.
Variable Adjustment: The model allows for variable redefinition to accommodate re-runs, such as adjusting for rejections and reassignments.
Requirements
Python 3.x
Gurobi Optimizer (for solving the optimization problem)
pandas (for data manipulation)
numpy (for mathematical operations)
Setup
Install Gurobi (if you don't have it already):

Download and install from Gurobi's website.
You’ll need a Gurobi license (a free academic license is available).
Install required Python libraries:

bash
Copy code
pip install pandas numpy gurobipy
Ensure you have all necessary data (student, studio, and instructor information) in a compatible format (e.g., CSV or Excel files).

Usage
1. Define the Optimization Model
To start the optimization process, initialize the model class and load your data (student info, studio info, etc.):

python
Copy code
from gurobipy import Model

# Initialize the model
model = ULAPlacementModel(num_ulas, num_studios, student_info, studio_info, instructor_info)

# Define variables and constraints
model.define_hiring_variables()
model.add_constraints()
2. Solve the Optimization Problem
Once the model is set up, you can solve it using the following command:

python
Copy code
model.m.optimize()
This will trigger the solver to find the optimal assignment of ULAs to studios while respecting the constraints.

3. Save and Load Model State
Save the current model to a file for later re-use:

python
Copy code
model.m.write('model_solution.sol')
Load a previously saved model to resume or modify the solution:

python
Copy code
model.m.read('model_solution.sol')
4. Handle Rejections and Rerun
If some ULAs have been rejected (or if the assignment needs to be updated), you can modify the model and rerun the optimization to fill remaining spots:

python
Copy code
# Update model (e.g., remove rejected ULAs, fill unfilled spots)
model.update_model_for_rerun()

# Re-run optimization
model.m.optimize()
5. Extract Results
Once the model has been solved, you can retrieve the variable values to see the assignments:

python
Copy code
# Retrieve variable values (e.g., ULA assignments)
assignments = model.get_assignments()

# Optionally, save results to a file
assignments.to_csv('ula_assignments.csv')
Model Structure
Variables:
x[i, s]: Binary variable indicating if ULA i is assigned to studio s.
alpha[i, j]: Binary variable indicating if ULA i is assigned to instructor j.
w[j, g]: Continuous variable for instructor gender assignments.
y[j, r]: Continuous variable for instructor race assignments.
z[j, m]: Continuous variable for instructor major assignments.
Constraints:
Studio Constraints: Ensure each studio has the required number of ULAs.
Instructor Constraints: Ensure each ULA is assigned to at most one instructor, with appropriate gender, race, and major representation.
E-campus Constraints: Ensure e-campus and non-e-campus students are assigned to the correct studio types.
Availability Constraints: Ensure ULA availability aligns with studio assignments.
Demographic Constraints: Ensure fairness by enforcing constraints based on ULA race, major, and gender.
