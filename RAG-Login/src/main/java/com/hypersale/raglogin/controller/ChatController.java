package com.hypersale.raglogin.controller;

import com.hypersale.raglogin.model.AppUser;
import com.hypersale.raglogin.model.RagMetric;
import com.hypersale.raglogin.model.SupportTicket;
import com.hypersale.raglogin.model.TicketStatus;
import com.hypersale.raglogin.repository.RagMetricRepository;
import com.hypersale.raglogin.repository.SupportTicketRepository;
import com.hypersale.raglogin.service.DocumentAccessService;
import com.hypersale.raglogin.service.RagClient;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/chat")
public class ChatController {
    private final RagClient rag;
    private final DocumentAccessService access;
    private final RagMetricRepository metrics;
    private final SupportTicketRepository tickets;

    public ChatController(RagClient rag, DocumentAccessService access, RagMetricRepository metrics, SupportTicketRepository tickets) {
        this.rag = rag; this.access = access; this.metrics = metrics; this.tickets = tickets;
    }

    @PostMapping("/query")
    public Map<String, Object> query(@AuthenticationPrincipal AppUser user, @Valid @RequestBody QueryRequest request) {
        Instant started = Instant.now();
        List<String> sources = access.accessible(user).stream().map(document -> document.getStoredPath()).toList();
        Map<String, Object> result = rag.query(user.getTenantId(), user.getRole().name(), request.question(),
                Math.max(1, Math.min(20, request.topKValue())), request.history() == null ? List.of() : request.history(), sources);
        List< Map<String, Object> > citations = (List<Map<String, Object>>) result.getOrDefault("citations", List.of());
        Map<String, String> documentIdsByPath = access.accessible(user).stream()
                .collect(java.util.stream.Collectors.toMap(document -> document.getStoredPath(), document -> document.getId(), (first, second) -> first));
        citations.forEach(citation -> citation.put("document_id", documentIdsByPath.get(citation.get("source"))));
        int evidenceCount = ((List<?>) result.getOrDefault("citations", List.of())).size();
        RagMetric metric = new RagMetric();
        metric.setTenantId(user.getTenantId()); metric.setUserId(user.getId());
        metric.setTraceId(String.valueOf(result.getOrDefault("trace_id", "")));
        metric.setEvidenceCount(evidenceCount);
        metric.setElapsedMs(Duration.between(started, Instant.now()).toMillis());
        metrics.save(metric);
        boolean needsHuman = Boolean.TRUE.equals(result.get("need_human_review")) || evidenceCount == 0;
        if (needsHuman) {
            SupportTicket ticket = new SupportTicket();
            ticket.setTenantId(user.getTenantId()); ticket.setUserId(user.getId());
            ticket.setTraceId(metric.getTraceId()); ticket.setQuestion(request.question());
            ticket.setSuggestedAnswer(String.valueOf(result.getOrDefault("answer", "")));
            ticket.setStatus(TicketStatus.OPEN); tickets.save(ticket);
            result = new java.util.LinkedHashMap<>(result);
            result.put("human_ticket_created", true);
            result.put("human_ticket_message", "知识库没有足够证据，已转人工客服处理");
        }
        return result;
    }

    public record QueryRequest(@NotBlank String question, Integer topK, List<Map<String, String>> history) {
        public int topKValue() { return topK == null ? 5 : topK; }
    }
}
