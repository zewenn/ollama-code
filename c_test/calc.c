#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Token types
typedef enum {
    TOKEN_NUMBER,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_MULTIPLY,
    TOKEN_DIVIDE,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_EOF
} TokenType;

// Token structure
typedef struct Token {
    TokenType type;
    char *value;
} Token;

// Parser state
typedef struct Parser {
    char *input;
    int pos;
    Token *current_token;
} Parser;

// Function prototypes
Token *tokenize(char *input);
void parser_init(Parser *parser, char *input);
void parser_next(Parser *parser);
void parser_error(Parser *parser, const char *message);
int parse_expression(Parser *parser);
int parse_term(Parser *parser);
int parse_factor(Parser *parser);

// Tokenize input string
Token *tokenize(char *input) {
    int len = strlen(input);
    Token *tokens = (Token *)malloc(100 * sizeof(Token));
    int token_count = 0;

    for (int i = 0; i < len; i++) {
        char c = input[i];
        if (c == ' ') continue;

        if (isdigit(c)) {
            char *num = (char *)malloc(10 * sizeof(char));
            int j = 0;
            while (i < len && isdigit(input[i])) {
                num[j++] = input[i++];
            }
            num[j] = '\0';
            tokens[token_count++] = (Token){TOKEN_NUMBER, num};
        } else if (c == '+' || c == '-') {
            tokens[token_count++] = (Token){(c == '+') ? TOKEN_PLUS : TOKEN_MINUS, NULL};
            i--; // Move to next character
        } else if (c == '*' || c == '/') {
            tokens[token_count++] = (Token){(c == '*') ? TOKEN_MULTIPLY : TOKEN_DIVIDE, NULL};
            i--; // Move
        } else if (c == '(') {
            tokens[token_count++] = (Token){TOKEN_LPAREN, NULL};
        } else if (c == ')') {
            tokens[token_count++] = (Token){TOKEN_RPAREN, NULL};
        } else {
            parser_error(NULL, "Unexpected character: %c", c);
        }
    }

    tokens[token_count].type = TOKEN_EOF;
    tokens[token_count].value = NULL;
    return tokens;
}

// Initialize parser
void parser_init(Parser *parser, char *input) {
    parser->input = input;
    parser->pos = 0;
    parser->current_token = NULL;
}

// Get next token
void parser_next(Parser *parser) {
    if (parser->current_token != NULL) {
        free(parser->current_token->value);
        free(parser->current_token);
    }

    if (parser->pos >= strlen(parser->input)) {
        parser->current_token = (Token *)malloc(sizeof(Token));
        parser->current_token->type = TOKEN_EOF;
        parser->current_token->value = NULL;
        return;
    }

    char c = parser->input[parser->pos];
    if (isdigit(c)) {
        int start = parser->pos;
        while (parser->pos < strlen(parser->input) && isdigit(parser->input[parser->pos])) {
            parser->pos++;
        }
        char *num = (char *)malloc((parser->pos - start + 1) * sizeof(char));
        strncpy(num, parser->input + start, parser->pos - start);
        num[parser->pos - start] = '\0';
        parser->current_token = (Token *)malloc(sizeof(Token));
        parser->current_token->type = TOKEN_NUMBER;
        parser->current_token->value = num;
    } else if (c == '+' || c == '-') {
        parser->current_token = (Token *)malloc(sizeof(Token));
        parser->current_token->type = (c == '+') ? TOKEN_PLUS : TOKEN_MINUS;
        parser->current_token->value = NULL;
        parser->pos++;
    } else if (c == '*' || c == '/') {
        parser->current_token = (Token *)malloc(sizeof(Token));
        parser->current_token->type = (c == '*') ? TOKEN_MULTIPLY : TOKEN_DIVIDE;
        parser->current_token->value = NULL;
        parser->pos++;
    } else if (c == '(') {
        parser->current_token = (Token *)malloc(sizeof(Token));
        parser->current_token->type = TOKEN_LPAREN;
        parser->current_token->value = NULL;
        parser->pos++;
    } else if (c == ')') {
        parser->current_token = (Token *)malloc(sizeof(Token));
        parser->current_token->type = TOKEN_RPAREN;
        parser->current_token->value = NULL;
        parser->pos++;
    } else {
        parser_error(parser, "Unexpected character: %c", c);
    }
}

// Parser error
void parser_error(Parser *parser, const char *message, ...) {
    va_list args;
    va_start(args, message);
    vfprintf(stderr, message, args);
    va_end(args);
    exit(EXIT_FAILURE);
}

// Parse expression
int parse_expression(Parser *parser) {
    int left = parse_term(parser);
    while (parser->current_token && parser->current_token->type == TOKEN_PLUS || parser->current_token->type == TOKEN_MINUS) {
        TokenType op = parser->current_token->type;
        parser_next(parser);
        int right = parse_term(parser);
        if (op == TOKEN_PLUS) left += right;
        else if (op == TOKEN_MINUS) left -= right;
    }
    return left;
}

// Parse term
int parse_term(Parser *parser) {
    int left = parse_factor(parser);
    while (parser->current_token && parser->current_token->type == TOKEN_MULTIPLY || parser->current_token->type == TOKEN_DIVIDE) {
        TokenType op = parser->current_token->type;
        parser_next(parser);
        int right = parse_factor(parser);
        if (op == TOKEN_MULTIPLY) left *= right;
        else if (op == TOKEN_DIVIDE) left /= right;
    }
    return left;
}

// Parse factor
int parse_factor(Parser *parser) {
    if (parser->current_token && parser->current_token->type == TOKEN_LPAREN) {
        parser_next(parser);
        int value = parse_expression(parser);
        if (parser->current_token && parser->current_token->type == TOKEN_RPAREN) {
            parser_next(parser);
        } else {
            parser_error(parser, "Expected closing parenthesis");
        }
        return value;
    } else if (parser->current_token && parser->current_token->type == TOKEN_NUMBER) {
        int value = atoi(parser->current_token->value);
        parser_next(parser);
        return value;
    } else {
        parser_error(parser, "Unexpected token");
    }
}

int main() {
    char input[100];
    printf("Enter expression: ");
    scanf("%s", input);
    Token *tokens = tokenize(input);
    Parser parser;
    parser_init(&parser, input);
    parser_next(&parser);
    int result = parse_expression(&parser);
    printf("Result: %d\n", result);
    return 0;
}
