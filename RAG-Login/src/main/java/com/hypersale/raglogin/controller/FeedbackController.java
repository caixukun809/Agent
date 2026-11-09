package com.hypersale.raglogin.controller;

import com.hypersale.raglogin.model.AnswerFeedback;
import com.hypersale.raglogin.model.AppUser;
import com.hypersale.raglogin.model.FeedbackStatus;
import com.hypersale.raglogin.model.FeedbackVote;
import com.hypersale.raglogin.repository.AnswerFeedbackRepository;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.Instant;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/feedback")
public class FeedbackController {
    private final AnswerFeedbackRepository feedback;

    public FeedbackController(AnswerFeedbackRepository feedback) { this.feedback = feedback; }

    @PostMapping
    public Map<String, Object> create(@AuthenticationPrincipal AppUser user, @Valid @RequestBody FeedbackRequest request) {
        AnswerFeedback item = new AnswerFeedback();
        item.setTenantId(user.getTenantId()); item.setUserId(user.getId()); item.setTraceId(request.traceId());
        item.setQuestion(request.question()); item.setAnswer(request.answer()); item.setVote(request.vote());
        item.setStatus(request.vote() == FeedbackVote.DOWN ? FeedbackStatus.PENDING : FeedbackStatus.APPROVED);
        feedback.save(item);
        return Map.of("id", item.getId(), "status", item.getStatus().name());
    }

    @GetMapping("/pending")
    @PreAuthorize("hasRole('ADMIN')")
    public List<AnswerFeedback> pending(@AuthenticationPrincipal AppUser user) {
        return feedback.findByTenantIdAndStatusOrderByCreatedAtDesc(user.getTenantId(), FeedbackStatus.PENDING);
    }

    @PatchMapping("/{id}")
    @PreAuthorize("hasRole('ADMIN')")
    public AnswerFeedback review(@AuthenticationPrincipal AppUser admin, @PathVariable Long id, @Valid @RequestBody ReviewRequest request) {
        if (request.status() == FeedbackStatus.PENDING) throw new IllegalArgumentException("审核状态必须是通过或拒绝");
        AnswerFeedback item = feedback.findById(id).filter(candidate -> candidate.getTenantId().equals(admin.getTenantId()))
                .orElseThrow(() -> new IllegalArgumentException("反馈不存在"));
        item.setStatus(request.status()); item.setReviewNote(request.note()); item.setReviewedBy(admin.getId()); item.setReviewedAt(Instant.now());
        return feedback.save(item);
    }

    public record FeedbackRequest(@NotBlank String traceId, @NotBlank String question, @NotBlank String answer,
                                  @jakarta.validation.constraints.NotNull FeedbackVote vote) {}
    public record ReviewRequest(FeedbackStatus status, String note) {}
}
