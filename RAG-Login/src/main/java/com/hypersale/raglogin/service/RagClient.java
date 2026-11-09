package com.hypersale.raglogin.service;

import com.hypersale.raglogin.config.RagProperties;
import tools.jackson.databind.ObjectMapper;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import java.util.List;
import java.util.Map;

@Service
public class RagClient {
    private static final ParameterizedTypeReference<Map<String, Object>> MAP_TYPE = new ParameterizedTypeReference<>() {};
    private final RestClient client;
    private final RagProperties properties;
    private final ObjectMapper mapper;
    public RagClient(RagProperties properties, ObjectMapper mapper) {
        this.client = RestClient.builder().baseUrl(properties.getApiUrl()).build();
        this.properties = properties;
        this.mapper = mapper;
    }

    public Map<String, Object> query(String tenantId, String role, String question, int topK,
                                     List<Map<String, String>> history, List<String> allowedSources) {
        Map<String, Object> request = Map.of(
                "tenant_id", tenantId,
                "role", role,
                "question", question,
                "top_k", topK,
                "history", history,
                "allowed_sources", allowedSources
        );
        return post("/api/query", request);
    }

    public Map<String, Object> ingestTenant(String tenantId) {
        return post("/api/tenant/ingest", Map.of("tenant_id", tenantId));
    }

    public Map<String, Object> preview(String tenantId, String source, int lineStart, int lineEnd) {
        return post("/api/tenant/preview", Map.of(
                "tenant_id", tenantId,
                "source", source,
                "line_start", lineStart,
                "line_end", lineEnd
        ));
    }

    public Map<String, Object> health() {
        try {
            return client.get().uri("/api/health").retrieve().body(MAP_TYPE);
        } catch (RestClientException exception) {
            return Map.of("status", "unavailable", "error", "RAG 服务连接失败");
        }
    }

    private Map<String, Object> post(String path, Map<String, Object> body) {
        try {
            Map<String, Object> response = client.post()
                    .uri(path)
                    .header("X-RAG-Internal-Key", properties.getInternalKey())
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(mapper.writeValueAsString(body))
                    .retrieve()
                    .onStatus(HttpStatusCode::isError, (request, result) -> {
                        String message = "RAG 服务返回 " + result.getStatusCode().value();
                        throw new RagUnavailableException(message);
                    })
                    .body(MAP_TYPE);
            return response == null ? Map.of() : response;
        } catch (RagUnavailableException exception) {
            throw exception;
        } catch (RestClientException exception) {
            throw new RagUnavailableException("RAG 服务不可用，请确认 Python 服务已启动");
        } catch (tools.jackson.core.JacksonException exception) {
            throw new RagUnavailableException("RAG 请求编码失败");
        }
    }

    public static class RagUnavailableException extends RuntimeException {
        public RagUnavailableException(String message) { super(message); }
    }
}
