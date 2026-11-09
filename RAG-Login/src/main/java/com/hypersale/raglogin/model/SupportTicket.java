package com.hypersale.raglogin.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Lob;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "support_tickets")
public class SupportTicket {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @Column(nullable = false, length = 80) private String tenantId;
    @Column(nullable = false) private Long userId;
    @Column(nullable = false, length = 64) private String traceId;
    @Lob @Column(nullable = false) private String question;
    @Lob private String suggestedAnswer;
    @Lob private String employeeAnswer;
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 12) private TicketStatus status;
    private Long assignedEmployeeId;
    @Column(nullable = false, updatable = false) private Instant createdAt;
    @Column(nullable = false) private Instant updatedAt;

    @PrePersist void created() { createdAt = updatedAt = Instant.now(); }
    @PreUpdate void updated() { updatedAt = Instant.now(); }
    public Long getId() { return id; }
    public String getTenantId() { return tenantId; }
    public void setTenantId(String tenantId) { this.tenantId = tenantId; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public String getTraceId() { return traceId; }
    public void setTraceId(String traceId) { this.traceId = traceId; }
    public String getQuestion() { return question; }
    public void setQuestion(String question) { this.question = question; }
    public String getSuggestedAnswer() { return suggestedAnswer; }
    public void setSuggestedAnswer(String suggestedAnswer) { this.suggestedAnswer = suggestedAnswer; }
    public String getEmployeeAnswer() { return employeeAnswer; }
    public void setEmployeeAnswer(String employeeAnswer) { this.employeeAnswer = employeeAnswer; }
    public TicketStatus getStatus() { return status; }
    public void setStatus(TicketStatus status) { this.status = status; }
    public Long getAssignedEmployeeId() { return assignedEmployeeId; }
    public void setAssignedEmployeeId(Long assignedEmployeeId) { this.assignedEmployeeId = assignedEmployeeId; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
}
