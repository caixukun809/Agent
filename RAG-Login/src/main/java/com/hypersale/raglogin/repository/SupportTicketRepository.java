package com.hypersale.raglogin.repository;

import com.hypersale.raglogin.model.SupportTicket;
import com.hypersale.raglogin.model.TicketStatus;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface SupportTicketRepository extends JpaRepository<SupportTicket, Long> {
    List<SupportTicket> findByTenantIdAndUserIdOrderByCreatedAtDesc(String tenantId, Long userId);
    List<SupportTicket> findByTenantIdAndStatusOrderByCreatedAtAsc(String tenantId, TicketStatus status);
    Optional<SupportTicket> findByTenantIdAndId(String tenantId, Long id);
    long countByTenantIdAndStatus(String tenantId, TicketStatus status);
}
