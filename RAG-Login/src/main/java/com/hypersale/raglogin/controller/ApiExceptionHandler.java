package com.hypersale.raglogin.controller;

import com.hypersale.raglogin.service.RagClient;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.server.ResponseStatusException;

import java.util.Map;

@RestControllerAdvice
public class ApiExceptionHandler {
    @ExceptionHandler(ResponseStatusException.class)
    ResponseEntity<Map<String, String>> status(ResponseStatusException exception) {
        return ResponseEntity.status(exception.getStatusCode()).body(Map.of("error", exception.getReason() == null ? "请求失败" : exception.getReason()));
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    ResponseEntity<Map<String, String>> invalid(MethodArgumentNotValidException exception) {
        String message = exception.getBindingResult().getFieldErrors().stream()
                .findFirst().map(error -> error.getDefaultMessage() == null ? "请求参数无效" : error.getDefaultMessage())
                .orElse("请求参数无效");
        return ResponseEntity.badRequest().body(Map.of("error", message));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    ResponseEntity<Map<String, String>> badRequest(IllegalArgumentException exception) {
        return ResponseEntity.badRequest().body(Map.of("error", exception.getMessage() == null ? "请求参数无效" : exception.getMessage()));
    }

    @ExceptionHandler(RagClient.RagUnavailableException.class)
    ResponseEntity<Map<String, String>> ragUnavailable(RagClient.RagUnavailableException exception) {
        return ResponseEntity.status(HttpStatus.BAD_GATEWAY).body(Map.of("error", exception.getMessage()));
    }
}
