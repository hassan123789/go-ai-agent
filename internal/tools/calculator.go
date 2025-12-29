package tools

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strconv"
)

// Calculator is a tool that evaluates mathematical expressions.
// It supports basic arithmetic operations: +, -, *, /, and parentheses.
type Calculator struct{}

// NewCalculator creates a new Calculator tool.
func NewCalculator() *Calculator {
	return &Calculator{}
}

// Name returns the tool name.
func (c *Calculator) Name() string {
	return "calculator"
}

// Description returns what this tool does.
func (c *Calculator) Description() string {
	return "Evaluates mathematical expressions. Supports +, -, *, /, parentheses, and decimal numbers. Example: '(2 + 3) * 4' returns '20'."
}

// Parameters returns the JSON Schema for the tool's input.
func (c *Calculator) Parameters() ParameterSchema {
	return ParameterSchema{
		Type: "object",
		Properties: map[string]PropertySchema{
			"expression": {
				Type:        "string",
				Description: "The mathematical expression to evaluate, e.g., '2 + 3 * 4'",
			},
		},
		Required: []string{"expression"},
	}
}

// calculatorArgs represents the arguments for the calculator tool.
type calculatorArgs struct {
	Expression string `json:"expression"`
}

// Execute evaluates the mathematical expression.
func (c *Calculator) Execute(_ context.Context, arguments string) (Result, error) {
	args, err := ParseArguments[calculatorArgs](arguments)
	if err != nil {
		return Failure("Invalid arguments: " + err.Error()), nil
	}

	if args.Expression == "" {
		return Failure("Expression cannot be empty"), nil
	}

	result, err := evaluateExpression(args.Expression)
	if err != nil {
		return Failure("Evaluation error: " + err.Error()), nil
	}

	return Success(formatNumber(result)), nil
}

// evaluateExpression parses and evaluates a mathematical expression.
// Uses Go's AST parser to safely parse expressions.
func evaluateExpression(expr string) (float64, error) {
	// Parse the expression as a Go expression
	node, err := parser.ParseExpr(expr)
	if err != nil {
		return 0, fmt.Errorf("invalid expression: %w", err)
	}

	return evalNode(node)
}

// evalNode recursively evaluates an AST node.
func evalNode(node ast.Expr) (float64, error) {
	switch n := node.(type) {
	case *ast.BasicLit:
		// Number literal
		return parseNumber(n.Value)

	case *ast.UnaryExpr:
		// Unary operator (e.g., -5)
		operand, err := evalNode(n.X)
		if err != nil {
			return 0, err
		}
		switch n.Op {
		case token.SUB:
			return -operand, nil
		case token.ADD:
			return operand, nil
		default:
			return 0, fmt.Errorf("unsupported unary operator: %s", n.Op)
		}

	case *ast.BinaryExpr:
		// Binary operator (e.g., 2 + 3)
		left, err := evalNode(n.X)
		if err != nil {
			return 0, err
		}
		right, err := evalNode(n.Y)
		if err != nil {
			return 0, err
		}

		switch n.Op {
		case token.ADD:
			return left + right, nil
		case token.SUB:
			return left - right, nil
		case token.MUL:
			return left * right, nil
		case token.QUO:
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			return left / right, nil
		default:
			return 0, fmt.Errorf("unsupported operator: %s", n.Op)
		}

	case *ast.ParenExpr:
		// Parenthesized expression
		return evalNode(n.X)

	default:
		return 0, fmt.Errorf("unsupported expression type: %T", node)
	}
}

// parseNumber parses a string as a number.
func parseNumber(s string) (float64, error) {
	// Try parsing as float
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid number: %s", s)
	}
	return f, nil
}

// formatNumber formats a float64 as a string, removing trailing zeros.
func formatNumber(f float64) string {
	// Check if it's an integer
	if f == float64(int64(f)) {
		return strconv.FormatInt(int64(f), 10)
	}
	return strconv.FormatFloat(f, 'f', -1, 64)
}
