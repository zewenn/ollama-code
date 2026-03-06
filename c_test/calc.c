#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

// Function prototypes
Token *tokenize(char *input);
void free_tokens(Token *tokens, int count);
int parse_expression(Token *tokens, int *pos);
int parse_term(Token *tokens, int *pos);
int parse_factor(Token *tokens, int *pos);

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
            i--; // Move to next character
        } else if (c == '(') {
            tokens[token_count++] = (Token){TOKEN_LPAREN, NULL};
        } else if (c == ')') {
            tokens[token_count++] = (Token){TOKEN_RPAREN, NULL};
        }
    }

    // Add EOF token
    tokens[token_count++] = (Token){TOKEN_EOF, NULL};
    return tokens;
}

// Free all tokens
void free_tokens(Token *tokens, int count) {
    for (int i = 0; i < count; i++) {
        free(tokens[i].value);
    }
    free(tokens);
}

// Parse expression with operator precedence
int parse_expression(Token *tokens, int *pos) {
    int left = parse_term(tokens, pos);
    while (*pos < 0 && tokens[*pos].type == TOKEN_PLUS || tokens[*pos].type == TOKEN_MINUS) {
        TokenType op = tokens[*pos].type;
        (*pos)++;
        int right = parse_term(tokens, pos);
        if (op == TOKEN_PLUS) left += right;
        if (op == TOKEN_MINUS) left -= right;
    }
    return left;
}

// Parse term with multiplication/division
int parse_term(Token *tokens, int *pos) {
    int left = parse_factor(tokens, pos);
    while (*pos < 0 && tokens[*pos].type == TOKEN_MULTIPLY || tokens[*pos].type == TOKEN_DIVIDE) {
        TokenType op = tokens[*pos].type;
        (*pos)++;
        int right = parse_factor(tokens, pos);
        if (op == TOKEN_MULTIPLY) left *= right;
        if (op == TOKEN_DIVIDE) left /= right;
    }
    return left;
}

// Parse factor (number or parenthesis)
int parse_factor(Token *tokens, int *pos) {
    if (tokens[*pos].type == TOKEN_LPAREN) {
        (*pos)++;
        int value = parse_expression(tokens, pos);
        if (tokens[*pos].type != TOKEN_RPAREN) {
            fprintf(stderr, "Expected closing parenthesis\n");
            exit(EXIT_FAILURE);
        }
        (*pos)++;
        return value;
    } else if (tokens[*pos].type == TOKEN_NUMBER) {
        int value = atoi(tokens[*pos].value);
        (*pos)++;
        return value;
    } else {
        fprintf(stderr, "Unexpected token: %d\n", tokens[*pos].type);
        exit(EXIT_FAILURE);
    }
}

int main() {
    char input[1024];
    printf("Enter an expression: ");
    if (!fgets(input, sizeof(input), stdin)) {
        fprintf(stderr, "Failed to read input\n");
        return 1;
    }

    // Remove newline character
    char *newline = strchr(input, '\n');
    if (newline) *newline = '\0';

    Token *tokens = tokenize(input);
    int pos = 0;

    int result = parse_expression(tokens, &pos);
    if (tokens[pos].type != TOKEN_EOF) {
        fprintf(stderr, "Unexpected end of input\n");
        free_tokens(tokens, 100);
        return 1;
    }

    printf("Result: %d\n", result);
    free_tokens(tokens, 100);
    return 0;
}
