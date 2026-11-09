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
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "answer_feedback")
public class AnswerFeedback {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @Column(nullable = false, length = 80) private String tenantId;
    @Column(nullable = false) private Long userId;
    @Column(nullable = false, length = 64) private String traceId;
    @Lob @Column(nullable = false) private String question;
    @Lob @Column(nullable = false) private String answer;
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 12) private FeedbackVote vote;
    @Enumerated(EnumType.STRING) @Column(nullable = false, length = 12) private FeedbackStatus status;
    @Column(length = 1000) private String reviewNote;
    private Long reviewedBy;
    private Instant reviewedAt;
    @Column(nullable = false, updatable = false) private Instant createdAt;

    @PrePersist void created() { createdAt = Instant.now(); }
    public Long getId() { return id; }
    public String getTenantId() { return tenantId; }
    public void setTenantId(String tenantId) { this.tenantId = tenantId; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public String getTraceId() { return traceId; }
    public void setTraceId(String traceId) { this.traceId = traceId; }
    public String getQuestion() { return question; }
    public void setQuestion(String question) { this.question = question; }
    public String getAnswer() { return answer; }
    public void setAnswer(String answer) { this.answer = answer; }
    public FeedbackVote getVote() { return vote; }
    public void setVote(FeedbackVote vote) { this.vote = vote; }
    public FeedbackStatus getStatus() { return status; }
    public void setStatus(FeedbackStatus status) { this.status = status; }
    public String getReviewNote() { return reviewNote; }
    public void setReviewNote(String reviewNote) { this.reviewNote = reviewNote; }
    public Long getReviewedBy() { return reviewedBy; }
    public void setReviewedBy(Long reviewedBy) { this.reviewedBy = reviewedBy; }
    public Instant getReviewedAt() { return reviewedAt; }
    public void setReviewedAt(Instant reviewedAt) { this.reviewedAt = reviewedAt; }
    public Instant getCreatedAt() { return createdAt; }
}
