package com.hypersale.raglogin.repository;

import com.hypersale.raglogin.model.AnswerFeedback;
import com.hypersale.raglogin.model.FeedbackStatus;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface AnswerFeedbackRepository extends JpaRepository<AnswerFeedback, Long> {
    List<AnswerFeedback> findByTenantIdAndStatusOrderByCreatedAtDesc(String tenantId, FeedbackStatus status);
    long countByTenantIdAndVote(String tenantId, com.hypersale.raglogin.model.FeedbackVote vote);
    long countByTenantIdAndStatus(String tenantId, FeedbackStatus status);
}
