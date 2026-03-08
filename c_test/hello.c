#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

int main() {
    FILE *file = fopen("greetings.txt", "r");
    if (file == NULL) {
        perror("Failed to open greetings.txt");
        return 1;
    }

    // Count lines
    int lineCount = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        lineCount++;
    }

    // Rewind and read all lines
    rewind(file);
    char **greetings = malloc(lineCount * sizeof(char *));
    for (int i = 0; i < lineCount; i++) {
        fgets(buffer, sizeof(buffer), file);
        greetings[i] = strdup(buffer);
    }

    // Get terminal width
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
        w.ws_col = 80; // Default width
    }

    // Select random greeting
    srand(time(NULL));
    int index = rand() % lineCount;
    char *greeting = greetings[index];
    int len = strlen(greeting);
    int padding = (w.ws_col - len) / 2;

    if (padding < 0) {
        printf("%s\n", greeting);
    } else {
        char spaces[padding + 1];
        memset(spaces, ' ', padding);
        spaces[padding] = '\0';
        printf("\n\n%s%s\n\n", spaces, greeting);
    }

    // Clean up
    for (int i = 0; i < lineCount; i++) {
        free(greetings[i]);
    }
    free(greetings);
    fclose(file);
    return 0;
}