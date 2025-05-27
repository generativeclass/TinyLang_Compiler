#!/usr/bin/env python3
"""
Enhanced TinyLang Compiler with Complete 6 Phases

Phases:
1. Lexical Analysis (Tokenization)
2. Syntax Analysis (Parsing)
3. Semantic Analysis
4. Intermediate Code Generation
5. Optimization
6. Code Generation
"""

import re
import sys

# Phase 1: Lexical Analysis (Tokenization)
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
                    if token_type != 'WHITESPACE':
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

# Phase 2: Syntax Analysis (Parsing)
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
        self.value_type = value_type

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

# Phase 3: Semantic Analysis
class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = set()
        self.errors = []

    def analyze(self, node):
        method_name = f'analyze_{node.__class__.__name__}'
        method = getattr(self, method_name, self.analyze_unknown)
        method(node)
        return self.errors

    def analyze_unknown(self, node):
        self.errors.append(f"Unknown node type: {node.__class__.__name__}")

    def analyze_Program(self, node):
        for statement in node.statements:
            self.analyze(statement)

    def analyze_VarDeclaration(self, node):
        if node.name in self.symbol_table:
            self.errors.append(f"Redeclaration of variable '{node.name}'")
        self.symbol_table.add(node.name)
        if node.initializer:
            self.analyze(node.initializer)

    def analyze_Assignment(self, node):
        if node.name not in self.symbol_table:
            self.errors.append(f"Assignment to undeclared variable '{node.name}'")
        self.analyze(node.value)

    def analyze_Identifier(self, node):
        if node.name not in self.symbol_table:
            self.errors.append(f"Use of undeclared variable '{node.name}'")

    def analyze_BinaryOperation(self, node):
        self.analyze(node.left)
        self.analyze(node.right)

    def analyze_UnaryOperation(self, node):
        self.analyze(node.operand)

    def analyze_PrintStatement(self, node):
        self.analyze(node.expression)

    def analyze_IfStatement(self, node):
        self.analyze(node.condition)
        self.analyze(node.then_branch)
        if node.else_branch:
            self.analyze(node.else_branch)

    def analyze_WhileStatement(self, node):
        self.analyze(node.condition)
        self.analyze(node.body)

    def analyze_Block(self, node):
        for statement in node.statements:
            self.analyze(statement)

    def analyze_Literal(self, node):
        pass  # Literals don't need semantic analysis

# Phase 4: Intermediate Code Generation
class IntermediateCodeGenerator:
    def __init__(self):
        self.code = []
        self.temp_counter = 0

    def generate(self, node):
        method_name = f'generate_{node.__class__.__name__}'
        method = getattr(self, method_name, self.generate_unknown)
        method(node)
        return self.code

    def generate_unknown(self, node):
        self.code.append(f"# Unknown node: {node.__class__.__name__}")

    def new_temp(self):
        temp = f"t{self.temp_counter}"
        self.temp_counter += 1
        return temp

    def generate_Program(self, node):
        for statement in node.statements:
            self.generate(statement)

    def generate_VarDeclaration(self, node):
        if node.initializer:
            temp = self.generate_expression(node.initializer)
            self.code.append(f"STORE {node.name}, {temp}")

    def generate_Assignment(self, node):
        temp = self.generate_expression(node.value)
        self.code.append(f"STORE {node.name}, {temp}")

    def generate_PrintStatement(self, node):
        temp = self.generate_expression(node.expression)
        self.code.append(f"PRINT {temp}")

    def generate_IfStatement(self, node):
        condition_temp = self.generate_expression(node.condition)
        else_label = f"L{self.temp_counter}"
        end_label = f"L{self.temp_counter + 1}"
        self.temp_counter += 2
        
        self.code.append(f"IF_FALSE {condition_temp} GOTO {else_label}")
        self.generate(node.then_branch)
        self.code.append(f"GOTO {end_label}")
        self.code.append(f"LABEL {else_label}")
        if node.else_branch:
            self.generate(node.else_branch)
        self.code.append(f"LABEL {end_label}")

    def generate_WhileStatement(self, node):
        start_label = f"L{self.temp_counter}"
        end_label = f"L{self.temp_counter + 1}"
        self.temp_counter += 2
        
        self.code.append(f"LABEL {start_label}")
        condition_temp = self.generate_expression(node.condition)
        self.code.append(f"IF_FALSE {condition_temp} GOTO {end_label}")
        self.generate(node.body)
        self.code.append(f"GOTO {start_label}")
        self.code.append(f"LABEL {end_label}")

    def generate_Block(self, node):
        for statement in node.statements:
            self.generate(statement)

    def generate_expression(self, node):
        if isinstance(node, Identifier):
            temp = self.new_temp()
            self.code.append(f"LOAD {temp}, {node.name}")
            return temp
        elif isinstance(node, Literal):
            temp = self.new_temp()
            self.code.append(f"LOAD {temp}, #{node.value}")
            return temp
        elif isinstance(node, BinaryOperation):
            left_temp = self.generate_expression(node.left)
            right_temp = self.generate_expression(node.right)
            result_temp = self.new_temp()
            op = {
                'PLUS': 'ADD',
                'MINUS': 'SUB',
                'MULTIPLY': 'MUL',
                'DIVIDE': 'DIV',
                'EQUAL': 'EQ',
                'NOT_EQUAL': 'NEQ',
                'LESS_THAN': 'LT',
                'GREATER_THAN': 'GT',
                'LESS_EQUAL': 'LE',
                'GREATER_EQUAL': 'GE'
            }.get(node.operator, node.operator)
            self.code.append(f"{op} {result_temp}, {left_temp}, {right_temp}")
            return result_temp
        elif isinstance(node, UnaryOperation):
            operand_temp = self.generate_expression(node.operand)
            result_temp = self.new_temp()
            if node.operator == 'MINUS':
                self.code.append(f"NEG {result_temp}, {operand_temp}")
            return result_temp
        else:
            temp = self.new_temp()
            self.code.append(f"# Unknown expression: {node.__class__.__name__} -> {temp}")
            return temp

# Phase 5: Optimization
class Optimizer:
    def __init__(self):
        self.optimized_code = []

    def optimize(self, intermediate_code):
        # Simple constant folding and dead code elimination
        constants = {}
        i = 0
        while i < len(intermediate_code):
            line = intermediate_code[i]
            
            # Constant folding for assignments
            if line.startswith("LOAD t") and ", #" in line:
                parts = line.split()
                temp = parts[1][:-1]  # Remove comma
                value = parts[2][1:]  # Remove #
                constants[temp] = value
            
            # Constant propagation
            elif any(temp in line for temp in constants):
                for temp, value in constants.items():
                    line = line.replace(temp, f"#{value}")
                self.optimized_code.append(line)
                i += 1
                continue
            
            # Dead code elimination for unused temps
            elif line.startswith("LOAD t") and i+1 < len(intermediate_code):
                temp = line.split()[1][:-1]
                used = any(temp in l for l in intermediate_code[i+1:])
                if not used:
                    i += 1
                    continue
            
            self.optimized_code.append(line)
            i += 1
        
        return self.optimized_code

# Phase 6: Code Generation
class TargetCodeGenerator:
    def __init__(self):
        self.code = []
        self.register_pool = ['r0', 'r1', 'r2', 'r3']
        self.register_map = {}

    def generate(self, optimized_code):
        for line in optimized_code:
            if line.startswith("#"):
                self.code.append(line)
                continue
            
            if line.startswith("LOAD"):
                parts = line.split()
                dest = parts[1][:-1]  # Remove comma
                src = parts[2]
                
                if src.startswith("#"):  # Constant
                    reg = self.get_register(dest)
                    self.code.append(f"MOV {reg}, {src}")
                else:  # Variable
                    reg = self.get_register(dest)
                    self.code.append(f"LD {reg}, {src}")
            
            elif line.startswith("STORE"):
                parts = line.split()
                dest = parts[1][:-1]  # Remove comma
                src = parts[2]
                
                if src.startswith("t"):  # Temporary
                    reg = self.register_map.get(src)
                    if reg:
                        self.code.append(f"ST {dest}, {reg}")
                    else:
                        self.code.append(f"# Error: Unknown temp {src}")
                else:  # Constant
                    reg = self.get_register('temp')
                    self.code.append(f"MOV {reg}, {src}")
                    self.code.append(f"ST {dest}, {reg}")
            
            elif line.startswith(("ADD", "SUB", "MUL", "DIV", "EQ", "NEQ", "LT", "GT", "LE", "GE")):
                op = line.split()[0]
                dest = line.split()[1][:-1]  # Remove comma
                src1 = line.split()[2][:-1]  # Remove comma
                src2 = line.split()[3]
                
                reg_dest = self.get_register(dest)
                reg_src1 = self.get_register(src1) if src1.startswith("t") else src1
                reg_src2 = self.get_register(src2) if src2.startswith("t") else src2
                
                self.code.append(f"{op} {reg_dest}, {reg_src1}, {reg_src2}")
            
            elif line.startswith("PRINT"):
                src = line.split()[1]
                if src.startswith("t"):
                    reg = self.register_map.get(src)
                    if reg:
                        self.code.append(f"OUT {reg}")
                    else:
                        self.code.append(f"# Error: Unknown temp {src}")
                else:
                    self.code.append(f"OUT #{src}")
            
            elif line.startswith("IF_FALSE"):
                parts = line.split()
                cond = parts[1]
                label = parts[3]
                
                reg = self.register_map.get(cond)
                if reg:
                    self.code.append(f"BNZ {reg}, {label}")
                else:
                    self.code.append(f"# Error: Unknown temp {cond}")
            
            elif line.startswith("GOTO"):
                label = line.split()[1]
                self.code.append(f"JMP {label}")
            
            elif line.startswith("LABEL"):
                label = line.split()[1]
                self.code.append(f"{label}:")
            
            elif line.startswith("NEG"):
                parts = line.split()
                dest = parts[1][:-1]  # Remove comma
                src = parts[2]
                
                reg_dest = self.get_register(dest)
                reg_src = self.get_register(src) if src.startswith("t") else src
                
                self.code.append(f"NEG {reg_dest}, {reg_src}")
            
            else:
                self.code.append(f"# Unknown instruction: {line}")
        
        return self.code

    def get_register(self, temp):
        if temp in self.register_map:
            return self.register_map[temp]
        
        if self.register_pool:
            reg = self.register_pool.pop(0)
            self.register_map[temp] = reg
            return reg
        
        # Register spilling - reuse r0
        spilled = 'r0'
        for t, r in list(self.register_map.items()):
            if r == spilled:
                self.code.append(f"ST {t}, {spilled}")
                del self.register_map[t]
                break
        
        self.register_map[temp] = spilled
        return spilled

# AST Printer
class ASTPrinter:
    def __init__(self):
        self.output = []
        self.indent_level = 0

    def print_ast(self, node):
        method_name = f'print_{node.__class__.__name__}'
        method = getattr(self, method_name, self.print_unknown)
        method(node)
        return '\n'.join(self.output)

    def print_unknown(self, node):
        self._write(f"Unknown node type: {node.__class__.__name__}")

    def _write(self, text):
        self.output.append("  " * self.indent_level + text)

    def _indent(self):
        self.indent_level += 1

    def _dedent(self):
        self.indent_level -= 1

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


# Program Executor class
class ProgramExecutor:
    def __init__(self, optimized_code):
        self.code = optimized_code
        self.variables = {}
        self.output = []
        self.pc = 0  # Program counter
        self.labels = {}
        
        # First pass to collect labels
        for i, line in enumerate(self.code):
            if line.startswith("LABEL "):
                label = line.split()[1]
                self.labels[label] = i

    def execute(self):
        while self.pc < len(self.code):
            line = self.code[self.pc]
            self.pc += 1
            
            if line.startswith("#") or line.startswith("LABEL"):
                continue
            elif line.startswith("STORE"):
                parts = line.split()
                var = parts[1][:-1]  # Remove comma
                value = parts[2]
                
                if value.startswith("#"):
                    self.variables[var] = int(value[1:])
                elif value.startswith("t"):
                    # Should have been optimized away
                    self.variables[var] = 0
            elif line.startswith("LOAD"):
                parts = line.split()
                temp = parts[1][:-1]  # Remove comma
                src = parts[2]
                
                if src.startswith("#"):
                    self.variables[temp] = int(src[1:])
                else:
                    self.variables[temp] = self.variables.get(src, 0)
            elif line.startswith("PRINT"):
                src = line.split()[1]
                if src.startswith("#"):
                    self.output.append(src[1:])
                elif src.startswith("t"):
                    self.output.append(str(self.variables.get(src, 0)))
                else:
                    self.output.append(str(self.variables.get(src, 0)))
            elif line.startswith("IF_FALSE"):
                parts = line.split()
                cond = parts[1]
                label = parts[3]
                
                condition = self.variables.get(cond, 0)
                if not condition:
                    self.pc = self.labels[label]
            elif line.startswith("GOTO"):
                label = line.split()[1]
                self.pc = self.labels[label]
            elif line.startswith("ADD"):
                parts = line.split()
                dest = parts[1][:-1]
                src1 = parts[2][:-1]
                src2 = parts[3]
                
                val1 = self.variables.get(src1, 0) if src1.startswith("t") else int(src1[1:]) if src1.startswith("#") else self.variables.get(src1, 0)
                val2 = self.variables.get(src2, 0) if src2.startswith("t") else int(src2[1:]) if src2.startswith("#") else self.variables.get(src2, 0)
                
                self.variables[dest] = val1 + val2
            elif line.startswith("SUB"):
                parts = line.split()
                dest = parts[1][:-1]
                src1 = parts[2][:-1]
                src2 = parts[3]
                
                val1 = self.variables.get(src1, 0) if src1.startswith("t") else int(src1[1:]) if src1.startswith("#") else self.variables.get(src1, 0)
                val2 = self.variables.get(src2, 0) if src2.startswith("t") else int(src2[1:]) if src2.startswith("#") else self.variables.get(src2, 0)
                
                self.variables[dest] = val1 - val2
            elif line.startswith("MUL"):
                parts = line.split()
                dest = parts[1][:-1]
                src1 = parts[2][:-1]
                src2 = parts[3]
                
                val1 = self.variables.get(src1, 0) if src1.startswith("t") else int(src1[1:]) if src1.startswith("#") else self.variables.get(src1, 0)
                val2 = self.variables.get(src2, 0) if src2.startswith("t") else int(src2[1:]) if src2.startswith("#") else self.variables.get(src2, 0)
                
                self.variables[dest] = val1 * val2
            elif line.startswith("DIV"):
                parts = line.split()
                dest = parts[1][:-1]
                src1 = parts[2][:-1]
                src2 = parts[3]
                
                val1 = self.variables.get(src1, 0) if src1.startswith("t") else int(src1[1:]) if src1.startswith("#") else self.variables.get(src1, 0)
                val2 = self.variables.get(src2, 0) if src2.startswith("t") else int(src2[1:]) if src2.startswith("#") else self.variables.get(src2, 0)
                
                self.variables[dest] = val1 // val2 if val2 != 0 else 0
            elif line.startswith("EQ"):
                parts = line.split()
                dest = parts[1][:-1]
                src1 = parts[2][:-1]
                src2 = parts[3]
                
                val1 = self.variables.get(src1, 0) if src1.startswith("t") else int(src1[1:]) if src1.startswith("#") else self.variables.get(src1, 0)
                val2 = self.variables.get(src2, 0) if src2.startswith("t") else int(src2[1:]) if src2.startswith("#") else self.variables.get(src2, 0)
                
                self.variables[dest] = 1 if val1 == val2 else 0
            elif line.startswith("NEQ"):
                parts = line.split()
                dest = parts[1][:-1]
                src1 = parts[2][:-1]
                src2 = parts[3]
                
                val1 = self.variables.get(src1, 0) if src1.startswith("t") else int(src1[1:]) if src1.startswith("#") else self.variables.get(src1, 0)
                val2 = self.variables.get(src2, 0) if src2.startswith("t") else int(src2[1:]) if src2.startswith("#") else self.variables.get(src2, 0)
                
                self.variables[dest] = 1 if val1 != val2 else 0
            elif line.startswith("LT"):
                parts = line.split()
                dest = parts[1][:-1]
                src1 = parts[2][:-1]
                src2 = parts[3]
                
                val1 = self.variables.get(src1, 0) if src1.startswith("t") else int(src1[1:]) if src1.startswith("#") else self.variables.get(src1, 0)
                val2 = self.variables.get(src2, 0) if src2.startswith("t") else int(src2[1:]) if src2.startswith("#") else self.variables.get(src2, 0)
                
                self.variables[dest] = 1 if val1 < val2 else 0
            elif line.startswith("GT"):
                parts = line.split()
                dest = parts[1][:-1]
                src1 = parts[2][:-1]
                src2 = parts[3]
                
                val1 = self.variables.get(src1, 0) if src1.startswith("t") else int(src1[1:]) if src1.startswith("#") else self.variables.get(src1, 0)
                val2 = self.variables.get(src2, 0) if src2.startswith("t") else int(src2[1:]) if src2.startswith("#") else self.variables.get(src2, 0)
                
                self.variables[dest] = 1 if val1 > val2 else 0
            elif line.startswith("LE"):
                parts = line.split()
                dest = parts[1][:-1]
                src1 = parts[2][:-1]
                src2 = parts[3]
                
                val1 = self.variables.get(src1, 0) if src1.startswith("t") else int(src1[1:]) if src1.startswith("#") else self.variables.get(src1, 0)
                val2 = self.variables.get(src2, 0) if src2.startswith("t") else int(src2[1:]) if src2.startswith("#") else self.variables.get(src2, 0)
                
                self.variables[dest] = 1 if val1 <= val2 else 0
            elif line.startswith("GE"):
                parts = line.split()
                dest = parts[1][:-1]
                src1 = parts[2][:-1]
                src2 = parts[3]
                
                val1 = self.variables.get(src1, 0) if src1.startswith("t") else int(src1[1:]) if src1.startswith("#") else self.variables.get(src1, 0)
                val2 = self.variables.get(src2, 0) if src2.startswith("t") else int(src2[1:]) if src2.startswith("#") else self.variables.get(src2, 0)
                
                self.variables[dest] = 1 if val1 >= val2 else 0
            elif line.startswith("NEG"):
                parts = line.split()
                dest = parts[1][:-1]
                src = parts[2]
                
                val = self.variables.get(src, 0) if src.startswith("t") else int(src[1:]) if src.startswith("#") else self.variables.get(src, 0)
                self.variables[dest] = -val
        
        return "\n".join(self.output)

# Updated Main Compiler Function
def compile_source(source_code, filename="output"):
    try:
        # Phase 1: Lexical Analysis
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        with open(f"{filename}_1_lexical.txt", "w") as f:
            f.write("TOKENS:\n")
            for token in tokens:
                f.write(f"{token}\n")

        # Phase 2: Syntax Analysis
        parser = Parser(tokens)
        ast = parser.parse()
        ast_printer = ASTPrinter()
        ast_str = ast_printer.print_ast(ast)
        with open(f"{filename}_2_syntax.txt", "w") as f:
            f.write("ABSTRACT SYNTAX TREE:\n")
            f.write(ast_str)

        # Phase 3: Semantic Analysis
        analyzer = SemanticAnalyzer()
        errors = analyzer.analyze(ast)
        with open(f"{filename}_3_semantic.txt", "w") as f:
            f.write("SEMANTIC ANALYSIS:\n")
            if errors:
                f.write("Errors:\n")
                for error in errors:
                    f.write(f"- {error}\n")
            else:
                f.write("No semantic errors found\n")
            f.write("\nSymbol Table:\n")
            for var in analyzer.symbol_table:
                f.write(f"- {var}\n")

        # Phase 4: Intermediate Code Generation
        intermediate_gen = IntermediateCodeGenerator()
        intermediate_code = intermediate_gen.generate(ast)
        with open(f"{filename}_4_intermediate.txt", "w") as f:
            f.write("INTERMEDIATE CODE:\n")
            for line in intermediate_code:
                f.write(f"{line}\n")

        # Phase 5: Optimization
        optimizer = Optimizer()
        optimized_code = optimizer.optimize(intermediate_code)
        with open(f"{filename}_5_optimized.txt", "w") as f:
            f.write("OPTIMIZED CODE:\n")
            for line in optimized_code:
                f.write(f"{line}\n")

        # Phase 6: Target Code Generation
        target_gen = TargetCodeGenerator()
        target_code = target_gen.generate(optimized_code)
        with open(f"{filename}_6_target.txt", "w") as f:
            f.write("TARGET CODE:\n")
            for line in target_code:
                f.write(f"{line}\n")

        # Phase 7: Program Execution and Output
        executor = ProgramExecutor(optimized_code)
        program_output = executor.execute()
        with open(f"{filename}_7_output.txt", "w") as f:
            f.write("PROGRAM OUTPUT:\n")
            f.write(program_output)

        print(f"Compilation completed successfully. Output files:")
        for i in range(1, 8):
            phase_names = ['lexical', 'syntax', 'semantic', 'intermediate', 
                         'optimized', 'target', 'output']
            print(f"- {filename}_{i}_{phase_names[i-1]}.txt")
        
        print("\nProgram Output:")
        print(program_output)

    except Exception as e:
        print(f"Compilation failed: {e}")

def main():
    if len(sys.argv) > 1:
        # Compile from file
        with open(sys.argv[1], 'r') as f:
            source_code = f.read()
        base_name = sys.argv[1].split('.')[0]
        compile_source(source_code, base_name)
    else:
        # Interactive mode
        print("TinyLang Compiler (Interactive Mode)")
        print("Enter your program and press Ctrl+D (Unix) or Ctrl+Z (Windows) to compile")
        print("-----------------------------------")
        source_lines = []
        try:
            while True:
                line = input()
                source_lines.append(line)
        except EOFError:
            source_code = '\n'.join(source_lines)
            compile_source(source_code, "interactive")

if __name__ == "__main__":
    main()
