package com.hypersale.raglogin.controller;

import com.hypersale.raglogin.service.RagClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
public class HealthController {
    private final RagClient rag;
    public HealthController(RagClient rag) { this.rag = rag; }

    @GetMapping("/api/health")
    public Map<String, Object> health() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("status", "ready");
        result.put("service", "RAG-Login");
        result.put("rag", rag.health());
        return result;
    }
}
