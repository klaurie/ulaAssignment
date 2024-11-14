import argparse

from model.ula_model import ULAModel


# Set up argument parser for optional command-line input
parser = argparse.ArgumentParser(description="Run ULA Model with optional input file")
parser.add_argument(
    "-i", "--input", 
    type=str, 
    help="Specify the input Excel file (e.g., 'Sample_Input_Data 1.xlsx')"
)

# Parse arguments
args = parser.parse_args()
input_file = args.input if args.input else 'data/Sample_Input_Data 1.xlsx'

# Specify the names and locations of the output and error files
output_ULA_FileName = 'ULA_output 1.csv'
output_studio_FileName = 'studio_output 1.csv'
error_ULA_FileName = 'ULA_error 1.csv'
error_studio_FileName = 'studio_error.csv'

# Initialize the ULAModel with the specified input and output files
ula_model = ULAModel(input_file, output_ULA_FileName, output_studio_FileName, "ula_model", False)

# Build and solve the model
ula_model.build_model()
ula_model.solve()
ula_model.export_model()