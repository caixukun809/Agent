package com.hypersale.raglogin.controller;

import com.hypersale.raglogin.model.AppUser;
import com.hypersale.raglogin.model.SupportTicket;
import com.hypersale.raglogin.model.TicketStatus;
import com.hypersale.raglogin.repository.SupportTicketRepository;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/support")
public class SupportController {
    private final SupportTicketRepository tickets;

    public SupportController(SupportTicketRepository tickets) { this.tickets = tickets; }

    @GetMapping("/mine")
    public List<SupportTicket> mine(@AuthenticationPrincipal AppUser user) {
        return tickets.findByTenantIdAndUserIdOrderByCreatedAtDesc(user.getTenantId(), user.getId());
    }

    @GetMapping("/queue")
    @PreAuthorize("hasAnyRole('ADMIN', 'EMPLOYEE')")
    public List<SupportTicket> queue(@AuthenticationPrincipal AppUser user) {
        return tickets.findByTenantIdAndStatusOrderByCreatedAtAsc(user.getTenantId(), TicketStatus.OPEN);
    }

    @PatchMapping("/{id}/answer")
    @PreAuthorize("hasAnyRole('ADMIN', 'EMPLOYEE')")
    public SupportTicket answer(@AuthenticationPrincipal AppUser employee, @PathVariable Long id, @Valid @RequestBody AnswerRequest request) {
        SupportTicket ticket = tickets.findByTenantIdAndId(employee.getTenantId(), id)
                .orElseThrow(() -> new IllegalArgumentException("工单不存在"));
        ticket.setEmployeeAnswer(request.answer()); ticket.setAssignedEmployeeId(employee.getId()); ticket.setStatus(TicketStatus.ANSWERED);
        return tickets.save(ticket);
    }

    @PatchMapping("/{id}/close")
    public SupportTicket close(@AuthenticationPrincipal AppUser user, @PathVariable Long id) {
        SupportTicket ticket = tickets.findByTenantIdAndId(user.getTenantId(), id)
                .orElseThrow(() -> new IllegalArgumentException("工单不存在"));
        if (user.getRole().name().equals("USER") && !ticket.getUserId().equals(user.getId())) {
            throw new org.springframework.web.server.ResponseStatusException(org.springframework.http.HttpStatus.FORBIDDEN, "无权关闭该工单");
        }
        ticket.setStatus(TicketStatus.CLOSED);
        return tickets.save(ticket);
    }

    public record AnswerRequest(@NotBlank String answer) {}
}
