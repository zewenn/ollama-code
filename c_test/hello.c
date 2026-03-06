#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

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

    // Select random greeting
    srand(time(NULL));
    int index = rand() % lineCount;
    printf("%s\n", greetings[index]);

    // Free memory
    for (int i = 0; i < lineCount; i++) {
        free(greetings[i]);
    }
    free(greetings);

    fclose(file);
    return 0;
}