#!/usr/bin/env python3
"""
TinyLang Compiler

This is a minimalist compiler for the TinyLang language, which supports:
- Variable declarations and assignments
- Basic arithmetic operations
- Conditionals
- Print statements
- Simple loops

The compiler follows these stages:
1. Lexical Analysis (Tokenization)
2. Syntax Analysis (Parsing)
3. Code Generation

Example TinyLang program:
```
var x = 5;
var y = 10;
print x + y;
if (x < y) {
    print "x is less than y";
}
var i = 0;
while (i < 5) {
    print i;
    i = i + 1;
}
```
"""

import re
import sys

# PART 1: LEXICAL ANALYSIS (TOKENIZER)
# Token types
TOKEN_TYPES = {
    'VAR': r'var',
    'IF': r'if',
    'ELSE': r'else',
    'WHILE': r'while',
    'PRINT': r'print',
    'ID': r'[a-zA-Z_][a-zA-Z0-9_]*',
    'NUMBER': r'\d+',
    'STRING': r'"[^"]*"',
    'PLUS': r'\+',
    'MINUS': r'-',
    'MULTIPLY': r'\*',
    'DIVIDE': r'/',
    'ASSIGN': r'=',
    'EQUAL': r'==',
    'NOT_EQUAL': r'!=',
    'LESS_THAN': r'<',
    'GREATER_THAN': r'>',
    'LESS_EQUAL': r'<=',
    'GREATER_EQUAL': r'>=',
    'LPAREN': r'\(',
    'RPAREN': r'\)',
    'LBRACE': r'\{',
    'RBRACE': r'\}',
    'SEMICOLON': r';',
    'WHITESPACE': r'\s+'
}

class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f'Token({self.type}, {repr(self.value)}, line={self.line}, col={self.column})'


class Lexer:
    def __init__(self, source_code):
        self.source_code = source_code
        self.tokens = []
        self.position = 0
        self.line = 1
        self.column = 1

    def tokenize(self):
        while self.position < len(self.source_code):
            match = None
            for token_type, pattern in TOKEN_TYPES.items():
                regex = re.compile('^' + pattern)
                match = regex.match(self.source_code[self.position:])
                if match:
                    value = match.group(0)
                    if token_type != 'WHITESPACE':  # Skip whitespace
                        self.tokens.append(Token(token_type, value, self.line, self.column))
                    
                    # Update line and column counters
                    newlines = value.count('\n')
                    if newlines > 0:
                        self.line += newlines
                        self.column = len(value) - value.rindex('\n') if '\n' in value else 1
                    else:
                        self.column += len(value)
                    
                    self.position += len(value)
                    break
            
            if not match:
                raise SyntaxError(f"Unexpected character at line {self.line}, column {self.column}: {self.source_code[self.position]}")
        
        return self.tokens


# PART 2: SYNTAX ANALYSIS (PARSER)
class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements

class VarDeclaration(ASTNode):
    def __init__(self, name, initializer):
        self.name = name
        self.initializer = initializer

class Assignment(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class BinaryOperation(ASTNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

class UnaryOperation(ASTNode):
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand

class Identifier(ASTNode):
    def __init__(self, name):
        self.name = name

class Literal(ASTNode):
    def __init__(self, value, value_type):
        self.value = value
        self.value_type = value_type  # 'number', 'string', etc.

class PrintStatement(ASTNode):
    def __init__(self, expression):
        self.expression = expression

class IfStatement(ASTNode):
    def __init__(self, condition, then_branch, else_branch=None):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

class WhileStatement(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class Block(ASTNode):
    def __init__(self, statements):
        self.statements = statements


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0

    def parse(self):
        statements = []
        while not self.is_at_end():
            statements.append(self.statement())
        return Program(statements)

    def statement(self):
        if self.match('VAR'):
            return self.var_declaration()
        elif self.match('PRINT'):
            return self.print_statement()
        elif self.match('IF'):
            return self.if_statement()
        elif self.match('WHILE'):
            return self.while_statement()
        elif self.match('LBRACE'):
            return self.block()
        else:
            return self.expression_statement()

    def var_declaration(self):
        name = self.consume('ID', "Expect variable name after 'var'").value
        initializer = None
        
        if self.match('ASSIGN'):
            initializer = self.expression()
        
        self.consume('SEMICOLON', "Expect ';' after variable declaration")
        return VarDeclaration(name, initializer)

    def print_statement(self):
        value = self.expression()
        self.consume('SEMICOLON', "Expect ';' after print statement")
        return PrintStatement(value)

    def if_statement(self):
        self.consume('LPAREN', "Expect '(' after 'if'")
        condition = self.expression()
        self.consume('RPAREN', "Expect ')' after if condition")
        
        then_branch = self.statement()
        else_branch = None
        
        if self.match('ELSE'):
            else_branch = self.statement()
        
        return IfStatement(condition, then_branch, else_branch)

    def while_statement(self):
        self.consume('LPAREN', "Expect '(' after 'while'")
        condition = self.expression()
        self.consume('RPAREN', "Expect ')' after while condition")
        body = self.statement()
        
        return WhileStatement(condition, body)

    def block(self):
        statements = []
        
        while not self.check('RBRACE') and not self.is_at_end():
            statements.append(self.statement())
        
        self.consume('RBRACE', "Expect '}' after block")
        return Block(statements)

    def expression_statement(self):
        expr = self.expression()
        self.consume('SEMICOLON', "Expect ';' after expression")
        return expr

    def expression(self):
        return self.assignment()

    def assignment(self):
        expr = self.equality()
        
        if self.match('ASSIGN'):
            if isinstance(expr, Identifier):
                value = self.assignment()
                return Assignment(expr.name, value)
            else:
                raise SyntaxError("Invalid assignment target")
        
        return expr

    def equality(self):
        expr = self.comparison()
        
        while self.match('EQUAL', 'NOT_EQUAL'):
            operator = self.previous().type
            right = self.comparison()
            expr = BinaryOperation(expr, operator, right)
        
        return expr

    def comparison(self):
        expr = self.addition()
        
        while self.match('LESS_THAN', 'GREATER_THAN', 'LESS_EQUAL', 'GREATER_EQUAL'):
            operator = self.previous().type
            right = self.addition()
            expr = BinaryOperation(expr, operator, right)
        
        return expr

    def addition(self):
        expr = self.multiplication()
        
        while self.match('PLUS', 'MINUS'):
            operator = self.previous().type
            right = self.multiplication()
            expr = BinaryOperation(expr, operator, right)
        
        return expr

    def multiplication(self):
        expr = self.unary()
        
        while self.match('MULTIPLY', 'DIVIDE'):
            operator = self.previous().type
            right = self.unary()
            expr = BinaryOperation(expr, operator, right)
        
        return expr

    def unary(self):
        if self.match('MINUS'):
            operator = self.previous().type
            right = self.unary()
            return UnaryOperation(operator, right)
        
        return self.primary()

    def primary(self):
        if self.match('NUMBER'):
            return Literal(int(self.previous().value), 'number')
        
        if self.match('STRING'):
            # Remove the quotes
            value = self.previous().value[1:-1]
            return Literal(value, 'string')
        
        if self.match('ID'):
            return Identifier(self.previous().value)
        
        if self.match('LPAREN'):
            expr = self.expression()
            self.consume('RPAREN', "Expect ')' after expression")
            return expr
        
        raise SyntaxError(f"Unexpected token: {self.peek().value}")

    def match(self, *types):
        for type in types:
            if self.check(type):
                self.advance()
                return True
        return False

    def check(self, type):
        if self.is_at_end():
            return False
        return self.peek().type == type

    def advance(self):
        if not self.is_at_end():
            self.current += 1
        return self.previous()

    def is_at_end(self):
        return self.current >= len(self.tokens)

    def peek(self):
        return self.tokens[self.current]

    def previous(self):
        return self.tokens[self.current - 1]

    def consume(self, type, error_message):
        if self.check(type):
            return self.advance()
        
        token = self.peek()
        raise SyntaxError(f"{error_message} at line {token.line}, column {token.column}")


# PART 3: CODE GENERATION
class Interpreter:
    def __init__(self, program):
        self.program = program
        self.environment = {}  # Variables storage
        self.execution_trace = []  # To store execution steps

    def interpret(self):
        self.execution_trace.append("Program execution started")
        for statement in self.program.statements:
            self.execute(statement)
        self.execution_trace.append("Program execution completed")
        
        # Write execution trace to file
        with open("execution_trace.txt", "w") as trace_file:
            for step in self.execution_trace:
                trace_file.write(step + "\n")
        print("Execution trace saved to execution_trace.txt")

    def execute(self, statement):
        # Use visitor pattern based on statement type
        method_name = f"execute_{statement.__class__.__name__}"
        method = getattr(self, method_name, self.execute_unknown)
        result = method(statement)
        self.execution_trace.append(f"Executed {statement.__class__.__name__}")
        return result

    def execute_unknown(self, statement):
        self.execution_trace.append(f"Unknown statement: {statement.__class__.__name__}")
        raise RuntimeError(f"Unknown statement type: {statement.__class__.__name__}")

    def execute_VarDeclaration(self, statement):
        value = None
        if statement.initializer:
            value = self.evaluate(statement.initializer)
        self.environment[statement.name] = value
        self.execution_trace.append(f"Declared variable '{statement.name}' with value {value}")

    def execute_PrintStatement(self, statement):
        value = self.evaluate(statement.expression)
        print(value)
        self.execution_trace.append(f"Printed: {value}")

    def execute_IfStatement(self, statement):
        condition_result = self.evaluate(statement.condition)
        self.execution_trace.append(f"If condition evaluated to: {condition_result}")
        
        if self.is_truthy(condition_result):
            self.execution_trace.append("Taking 'then' branch")
            self.execute(statement.then_branch)
        elif statement.else_branch:
            self.execution_trace.append("Taking 'else' branch")
            self.execute(statement.else_branch)
        else:
            self.execution_trace.append("Condition false, no 'else' branch")

    def execute_WhileStatement(self, statement):
        iterations = 0
        while True:
            condition_result = self.evaluate(statement.condition)
            self.execution_trace.append(f"While condition evaluated to: {condition_result} (iteration {iterations})")
            
            if not self.is_truthy(condition_result):
                break
                
            self.execute(statement.body)
            iterations += 1

    def execute_Block(self, statement):
        self.execution_trace.append("Entering code block")
        for stmt in statement.statements:
            self.execute(stmt)
        self.execution_trace.append("Exiting code block")

    def execute_Assignment(self, statement):
        value = self.evaluate(statement.value)
        if statement.name not in self.environment:
            self.execution_trace.append(f"Error: Undefined variable '{statement.name}'")
            raise RuntimeError(f"Undefined variable '{statement.name}'")
        
        self.environment[statement.name] = value
        self.execution_trace.append(f"Assigned {value} to '{statement.name}'")
        return value

    def execute_BinaryOperation(self, expression):
        return self.evaluate(expression)

    def evaluate(self, expression):
        # Use visitor pattern based on expression type
        method_name = f"evaluate_{expression.__class__.__name__}"
        method = getattr(self, method_name, self.evaluate_unknown)
        result = method(expression)
        self.execution_trace.append(f"Evaluated {expression.__class__.__name__} to {result}")
        return result

    def evaluate_unknown(self, expression):
        self.execution_trace.append(f"Unknown expression: {expression.__class__.__name__}")
        raise RuntimeError(f"Unknown expression type: {expression.__class__.__name__}")

    def evaluate_Literal(self, expression):
        return expression.value

    def evaluate_Identifier(self, expression):
        if expression.name not in self.environment:
            self.execution_trace.append(f"Error: Undefined variable '{expression.name}'")
            raise RuntimeError(f"Undefined variable '{expression.name}'")
        return self.environment[expression.name]

    def evaluate_BinaryOperation(self, expression):
        left = self.evaluate(expression.left)
        right = self.evaluate(expression.right)
        
        self.execution_trace.append(f"Binary operation: {left} {expression.operator} {right}")
        
        if expression.operator == 'PLUS':
            return left + right
        elif expression.operator == 'MINUS':
            return left - right
        elif expression.operator == 'MULTIPLY':
            return left * right
        elif expression.operator == 'DIVIDE':
            return left // right if isinstance(left, int) and isinstance(right, int) else left / right
        elif expression.operator == 'EQUAL':
            return left == right
        elif expression.operator == 'NOT_EQUAL':
            return left != right
        elif expression.operator == 'LESS_THAN':
            return left < right
        elif expression.operator == 'GREATER_THAN':
            return left > right
        elif expression.operator == 'LESS_EQUAL':
            return left <= right
        elif expression.operator == 'GREATER_EQUAL':
            return left >= right
        else:
            self.execution_trace.append(f"Error: Unknown binary operator: {expression.operator}")
            raise RuntimeError(f"Unknown binary operator: {expression.operator}")

    def evaluate_UnaryOperation(self, expression):
        right = self.evaluate(expression.operand)
        
        if expression.operator == 'MINUS':
            return -right
        else:
            self.execution_trace.append(f"Error: Unknown unary operator: {expression.operator}")
            raise RuntimeError(f"Unknown unary operator: {expression.operator}")

    def evaluate_Assignment(self, expression):
        value = self.evaluate(expression.value)
        self.environment[expression.name] = value
        self.execution_trace.append(f"Assignment expression: {expression.name} = {value}")
        return value

    def is_truthy(self, value):
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        return True


def compile_and_run(source_code, filename='program'):
    """
    Main entry point to compile and run TinyLang code
    Outputs separate files for each compilation stage
    """
    try:
        base_filename = filename.split('/')[-1].split('.')[0]  # Extract base name without extension
        
        # 1. Tokenize
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        # Save tokens to file
        with open(f"{base_filename}_tokens.txt", "w") as token_file:
            token_file.write("TOKEN LIST:\n")
            for i, token in enumerate(tokens):
                token_file.write(f"{i}: {token}\n")
        print(f"Tokens saved to {base_filename}_tokens.txt")
        
        # Save the original source code
        with open(f"{base_filename}_source.tl", "w") as source_file:
            source_file.write(source_code)
        print(f"Source code saved to {base_filename}_source.tl")
        
        # 2. Parse
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Save AST to file using a custom AST printer
        with open(f"{base_filename}_ast.txt", "w") as ast_file:
            ast_file.write("ABSTRACT SYNTAX TREE:\n")
            ast_printer = ASTPrinter(ast_file)
            ast_printer.print_ast(ast)
        print(f"AST saved to {base_filename}_ast.txt")
        
        # 3. Interpret (code generation step would be here in a real compiler)
        with open(f"{base_filename}_output.txt", "w") as output_file:
            # Redirect print output to file
            original_stdout = sys.stdout
            sys.stdout = output_file
            
            interpreter = Interpreter(ast)
            interpreter.interpret()
            
            # Restore original stdout
            sys.stdout = original_stdout
        print(f"Program output saved to {base_filename}_output.txt")
        
        # The execution trace is saved by the interpreter itself
        
        # Generate simple bytecode (for demonstration)
        bytecode = ByteCodeGenerator().generate(ast)
        with open(f"{base_filename}_bytecode.txt", "w") as bytecode_file:
            for i, instruction in enumerate(bytecode):
                bytecode_file.write(f"{i:04d}: {instruction}\n")
        print(f"Bytecode saved to {base_filename}_bytecode.txt")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


# Simple ByteCode Generator
class ByteCodeGenerator:
    def __init__(self):
        self.bytecode = []
        self.constants = []
        self.variables = {}
        self.label_counter = 0
    
    def generate(self, node):
        """Generate bytecode for the AST"""
        self.bytecode = []
        self._visit(node)
        self.bytecode.append(("HALT",))
        return self.bytecode
    
    def _visit(self, node):
        """Visit a node in the AST and generate bytecode"""
        method_name = f"_visit_{node.__class__.__name__}"
        method = getattr(self, method_name, self._visit_unknown)
        return method(node)
    
    def _visit_unknown(self, node):
        self.bytecode.append((f"UNKNOWN_NODE: {node.__class__.__name__}",))
    
    def _visit_Program(self, node):
        for statement in node.statements:
            self._visit(statement)
    
    def _visit_VarDeclaration(self, node):
        # Add variable to the variables dictionary
        if node.name not in self.variables:
            self.variables[node.name] = len(self.variables)
        
        # Generate code for initializer if it exists
        if node.initializer:
            self._visit(node.initializer)
            self.bytecode.append(("STORE_VAR", node.name))
        else:
            # Push a default value (0) and store it
            self.bytecode.append(("PUSH_CONST", 0))
            self.bytecode.append(("STORE_VAR", node.name))
    
    def _visit_PrintStatement(self, node):
        self._visit(node.expression)
        self.bytecode.append(("PRINT",))
    
    def _visit_IfStatement(self, node):
        else_label = self._next_label()
        end_label = self._next_label()
        
        # Generate code for the condition
        self._visit(node.condition)
        self.bytecode.append(("JUMP_IF_FALSE", else_label))
        
        # Generate code for the then branch
        self._visit(node.then_branch)
        self.bytecode.append(("JUMP", end_label))
        
        # Add else label
        self.bytecode.append(("LABEL", else_label))
        
        # Generate code for the else branch if it exists
        if node.else_branch:
            self._visit(node.else_branch)
        
        # Add end label
        self.bytecode.append(("LABEL", end_label))
    
    def _visit_WhileStatement(self, node):
        start_label = self._next_label()
        end_label = self._next_label()
        
        # Add start label
        self.bytecode.append(("LABEL", start_label))
        
        # Generate code for the condition
        self._visit(node.condition)
        self.bytecode.append(("JUMP_IF_FALSE", end_label))
        
        # Generate code for the body
        self._visit(node.body)
        
        # Jump back to the start
        self.bytecode.append(("JUMP", start_label))
        
        # Add end label
        self.bytecode.append(("LABEL", end_label))
    
    def _visit_Block(self, node):
        for statement in node.statements:
            self._visit(statement)
    
    def _visit_Assignment(self, node):
        self._visit(node.value)
        self.bytecode.append(("STORE_VAR", node.name))
    
    def _visit_BinaryOperation(self, node):
        # Visit the left and right operands
        self._visit(node.left)
        self._visit(node.right)
        
        # Generate the appropriate operation
        if node.operator == 'PLUS':
            self.bytecode.append(("ADD",))
        elif node.operator == 'MINUS':
            self.bytecode.append(("SUBTRACT",))
        elif node.operator == 'MULTIPLY':
            self.bytecode.append(("MULTIPLY",))
        elif node.operator == 'DIVIDE':
            self.bytecode.append(("DIVIDE",))
        elif node.operator == 'EQUAL':
            self.bytecode.append(("EQUAL",))
        elif node.operator == 'NOT_EQUAL':
            self.bytecode.append(("NOT_EQUAL",))
        elif node.operator == 'LESS_THAN':
            self.bytecode.append(("LESS_THAN",))
        elif node.operator == 'GREATER_THAN':
            self.bytecode.append(("GREATER_THAN",))
        elif node.operator == 'LESS_EQUAL':
            self.bytecode.append(("LESS_EQUAL",))
        elif node.operator == 'GREATER_EQUAL':
            self.bytecode.append(("GREATER_EQUAL",))
        else:
            self.bytecode.append((f"UNKNOWN_OPERATOR: {node.operator}",))
    
    def _visit_UnaryOperation(self, node):
        self._visit(node.operand)
        if node.operator == 'MINUS':
            self.bytecode.append(("NEGATE",))
        else:
            self.bytecode.append((f"UNKNOWN_UNARY_OPERATOR: {node.operator}",))
    
    def _visit_Identifier(self, node):
        self.bytecode.append(("LOAD_VAR", node.name))
    
    def _visit_Literal(self, node):
        self.bytecode.append(("PUSH_CONST", node.value))
    
    def _next_label(self):
        """Generate a unique label for jumps"""
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label


# AST Printer for visualization
class ASTPrinter:
    def __init__(self, file):
        self.file = file
        self.indent_level = 0
        
    def print_ast(self, node):
        method_name = f"print_{node.__class__.__name__}"
        method = getattr(self, method_name, self.print_unknown)
        method(node)
    
    def print_unknown(self, node):
        self._write(f"Unknown node type: {node.__class__.__name__}")
    
    def print_Program(self, node):
        self._write("Program:")
        self._indent()
        for statement in node.statements:
            self.print_ast(statement)
        self._dedent()
    
    def print_VarDeclaration(self, node):
        self._write(f"VarDeclaration: {node.name}")
        if node.initializer:
            self._indent()
            self._write("Initializer:")
            self._indent()
            self.print_ast(node.initializer)
            self._dedent()
            self._dedent()
    
    def print_Assignment(self, node):
        self._write(f"Assignment: {node.name} =")
        self._indent()
        self.print_ast(node.value)
        self._dedent()
    
    def print_BinaryOperation(self, node):
        self._write(f"BinaryOperation: {node.operator}")
        self._indent()
        self._write("Left:")
        self._indent()
        self.print_ast(node.left)
        self._dedent()
        self._write("Right:")
        self._indent()
        self.print_ast(node.right)
        self._dedent()
        self._dedent()
    
    def print_UnaryOperation(self, node):
        self._write(f"UnaryOperation: {node.operator}")
        self._indent()
        self.print_ast(node.operand)
        self._dedent()
    
    def print_Identifier(self, node):
        self._write(f"Identifier: {node.name}")
    
    def print_Literal(self, node):
        self._write(f"Literal: {repr(node.value)} ({node.value_type})")
    
    def print_PrintStatement(self, node):
        self._write("PrintStatement:")
        self._indent()
        self.print_ast(node.expression)
        self._dedent()
    
    def print_IfStatement(self, node):
        self._write("IfStatement:")
        self._indent()
        self._write("Condition:")
        self._indent()
        self.print_ast(node.condition)
        self._dedent()
        self._write("Then:")
        self._indent()
        self.print_ast(node.then_branch)
        self._dedent()
        if node.else_branch:
            self._write("Else:")
            self._indent()
            self.print_ast(node.else_branch)
            self._dedent()
        self._dedent()
    
    def print_WhileStatement(self, node):
        self._write("WhileStatement:")
        self._indent()
        self._write("Condition:")
        self._indent()
        self.print_ast(node.condition)
        self._dedent()
        self._write("Body:")
        self._indent()
        self.print_ast(node.body)
        self._dedent()
        self._dedent()
    
    def print_Block(self, node):
        self._write("Block:")
        self._indent()
        for statement in node.statements:
            self.print_ast(statement)
        self._dedent()
    
    def _write(self, text):
        self.file.write("  " * self.indent_level + text + "\n")
    
    def _indent(self):
        self.indent_level += 1
    
    def _dedent(self):
        self.indent_level -= 1


def main():
    if len(sys.argv) > 1:
        # Run the file provided as argument
        filename = sys.argv[1]
        with open(filename, 'r') as file:
            source = file.read()
            compile_and_run(source, filename)
    else:
        # Interactive mode
        print("TinyLang Interpreter (Press Ctrl+D to exit)")
        print("Type your program and press Enter twice to execute:")
        
        while True:
            try:
                lines = []
                print(">> ", end="")
                while True:
                    line = input()
                    if not line.strip():
                        break
                    lines.append(line)
                
                source = "\n".join(lines)
                if source.strip():
                    compile_and_run(source, 'interactive')
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nExiting...")
                break


if __name__ == "__main__":
    main()