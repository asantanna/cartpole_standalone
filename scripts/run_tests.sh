#!/bin/bash
# Simple test runner script for CartPole standalone tests

echo "Running CartPole Standalone Test Suite"
echo "======================================"

# Check if Isaac Gym is available
python -c "import isaacgym" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Isaac Gym is not installed!"
    echo "Please install Isaac Gym before running tests."
    exit 1
fi

# Install test dependencies if needed
if [ "$1" == "--install" ]; then
    echo "Installing test dependencies..."
    pip install -r requirements-test.txt
fi

# Run tests with different configurations
if [ "$1" == "--fast" ]; then
    echo "Running fast tests only (excluding slow tests)..."
    pytest -m "not slow" -v
elif [ "$1" == "--slow" ]; then
    echo "Running slow tests only..."
    pytest -m "slow" -v
elif [ "$1" == "--coverage" ]; then
    echo "Running tests with coverage..."
    pytest --cov=. --cov-report=html --cov-report=term -v
elif [ "$1" == "--specific" ] && [ -n "$2" ]; then
    echo "Running specific test: $2"
    pytest -v "$2"
else
    echo "Running all tests..."
    pytest -v
fi

# Print summary
echo ""
echo "Test run completed!"
echo ""
echo "Usage options:"
echo "  ./run_tests.sh              # Run all tests"
echo "  ./run_tests.sh --fast       # Run only fast tests"
echo "  ./run_tests.sh --slow       # Run only slow tests"
echo "  ./run_tests.sh --coverage   # Run with coverage report"
echo "  ./run_tests.sh --install    # Install test dependencies"
echo "  ./run_tests.sh --specific <test_file>  # Run specific test file"