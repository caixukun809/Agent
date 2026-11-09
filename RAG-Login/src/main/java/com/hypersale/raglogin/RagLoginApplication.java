package com.hypersale.raglogin;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import com.hypersale.raglogin.config.RagProperties;

@SpringBootApplication
@EnableConfigurationProperties(RagProperties.class)
public class RagLoginApplication {

    public static void main(String[] args) {
        loadProjectDotEnv();
        SpringApplication.run(RagLoginApplication.class, args);
    }

    private static void loadProjectDotEnv() {
        java.nio.file.Path workingDirectory = java.nio.file.Path.of("").toAbsolutePath().normalize();
        java.nio.file.Path[] candidates = {
                workingDirectory.resolve(".env"),
                workingDirectory.resolve("../../Agent/.env").normalize(),
                workingDirectory.resolve("../.env").normalize()
        };
        for (java.nio.file.Path candidate : candidates) {
            if (!java.nio.file.Files.isRegularFile(candidate)) continue;
            try {
                for (String rawLine : java.nio.file.Files.readAllLines(candidate)) {
                    String line = rawLine.trim();
                    if (line.isEmpty() || line.startsWith("#")) continue;
                    if (line.startsWith("export ")) line = line.substring(7).trim();
                    int separator = line.indexOf('=');
                    if (separator < 1) continue;
                    String key = line.substring(0, separator).trim();
                    String value = line.substring(separator + 1).trim();
                    if (value.length() >= 2 && ((value.startsWith("\"") && value.endsWith("\""))
                            || (value.startsWith("'") && value.endsWith("'")))) {
                        value = value.substring(1, value.length() - 1);
                    }
                    if (System.getenv(key) == null && System.getProperty(key) == null) {
                        System.setProperty(key, value);
                    }
                }
                return;
            } catch (java.io.IOException ignored) {
                return;
            }
        }
    }

}
