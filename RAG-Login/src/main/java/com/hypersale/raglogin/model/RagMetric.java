package com.hypersale.raglogin.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "rag_metrics")
public class RagMetric {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @Column(nullable = false, length = 80) private String tenantId;
    @Column(nullable = false) private Long userId;
    @Column(nullable = false, length = 64) private String traceId;
    @Column(nullable = false) private int evidenceCount;
    @Column(nullable = false) private long elapsedMs;
    @Column(nullable = false, updatable = false) private Instant createdAt;

    @PrePersist void created() { createdAt = Instant.now(); }
    public Long getId() { return id; }
    public String getTenantId() { return tenantId; }
    public void setTenantId(String tenantId) { this.tenantId = tenantId; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public String getTraceId() { return traceId; }
    public void setTraceId(String traceId) { this.traceId = traceId; }
    public int getEvidenceCount() { return evidenceCount; }
    public void setEvidenceCount(int evidenceCount) { this.evidenceCount = evidenceCount; }
    public long getElapsedMs() { return elapsedMs; }
    public void setElapsedMs(long elapsedMs) { this.elapsedMs = elapsedMs; }
    public Instant getCreatedAt() { return createdAt; }
}
