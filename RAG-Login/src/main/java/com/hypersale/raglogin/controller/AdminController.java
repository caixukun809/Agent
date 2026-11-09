package com.hypersale.raglogin.controller;

import com.hypersale.raglogin.config.RagProperties;
import com.hypersale.raglogin.model.AppUser;
import com.hypersale.raglogin.model.UserRole;
import com.hypersale.raglogin.model.Tenant;
import com.hypersale.raglogin.repository.AnswerFeedbackRepository;
import com.hypersale.raglogin.repository.AppUserRepository;
import com.hypersale.raglogin.repository.KnowledgeDocumentRepository;
import com.hypersale.raglogin.repository.RagMetricRepository;
import com.hypersale.raglogin.repository.SupportTicketRepository;
import com.hypersale.raglogin.repository.TenantRepository;
import com.hypersale.raglogin.service.RagClient;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

import java.util.Map;

@RestController
@RequestMapping("/api/admin")
public class AdminController {
    private final AppUserRepository users;
    private final PasswordEncoder passwords;
    private final RagMetricRepository metrics;
    private final AnswerFeedbackRepository feedback;
    private final SupportTicketRepository tickets;
    private final KnowledgeDocumentRepository documents;
    private final RagClient rag;
    private final RagProperties properties;
    private final TenantRepository tenants;

    public AdminController(AppUserRepository users, PasswordEncoder passwords, RagMetricRepository metrics,
                           AnswerFeedbackRepository feedback, SupportTicketRepository tickets,
                           KnowledgeDocumentRepository documents, RagClient rag, RagProperties properties, TenantRepository tenants) {
        this.users = users; this.passwords = passwords; this.metrics = metrics; this.feedback = feedback;
        this.tickets = tickets; this.documents = documents; this.rag = rag; this.properties = properties; this.tenants = tenants;
    }

    @GetMapping("/metrics")
    public Map<String, Object> metrics(@AuthenticationPrincipal AppUser admin) {
        String tenant = admin.getTenantId();
        long total = metrics.countByTenantId(tenant);
        long noEvidence = metrics.countByTenantIdAndEvidenceCount(tenant, 0);
        return Map.of("tenant_id", tenant, "queries", total, "with_evidence", total - noEvidence,
                "without_evidence", noEvidence, "positive_feedback", feedback.countByTenantIdAndVote(tenant, com.hypersale.raglogin.model.FeedbackVote.UP),
                "negative_feedback", feedback.countByTenantIdAndVote(tenant, com.hypersale.raglogin.model.FeedbackVote.DOWN),
                "pending_feedback", feedback.countByTenantIdAndStatus(tenant, com.hypersale.raglogin.model.FeedbackStatus.PENDING),
                "open_tickets", tickets.countByTenantIdAndStatus(tenant, com.hypersale.raglogin.model.TicketStatus.OPEN),
                "documents", documents.findByTenantIdOrderByCreatedAtDesc(tenant).size(), "rag_health", rag.health());
    }

    @GetMapping("/users")
    public Object users(@AuthenticationPrincipal AppUser admin) { return users.findByTenantIdOrderByCreatedAtDesc(admin.getTenantId()).stream().map(user -> Map.of(
            "id", user.getId(), "username", user.getUsername(), "display_name", user.getDisplayName(), "role", user.getRole(), "enabled", user.isEnabled())).toList(); }

    @PostMapping("/users")
    public Object createUser(@AuthenticationPrincipal AppUser admin, @Valid @RequestBody CreateUserRequest request) {
        if (users.existsByUsername(request.username())) throw new ResponseStatusException(HttpStatus.CONFLICT, "用户名已存在");
        AppUser user = new AppUser(); user.setUsername(request.username().trim()); user.setDisplayName(request.displayName().trim());
        user.setPasswordHash(passwords.encode(request.password())); user.setRole(request.role()); user.setTenantId(admin.getTenantId());
        AppUser saved = users.save(user);
        return Map.of("id", saved.getId(), "username", saved.getUsername(), "display_name", saved.getDisplayName(), "role", saved.getRole());
    }

    @PostMapping("/reindex")
    public Map<String, Object> reindex(@AuthenticationPrincipal AppUser admin) {
        Map<String, Object> result = rag.ingestTenant(admin.getTenantId());
        var tenantDocuments = documents.findByTenantIdOrderByCreatedAtDesc(admin.getTenantId());
        tenantDocuments.forEach(document -> document.setIndexStatus("INDEXED"));
        documents.saveAll(tenantDocuments);
        return result;
    }

    @GetMapping("/tenants")
    public Object tenants(@AuthenticationPrincipal AppUser admin) {
        requirePlatformAdmin(admin);
        return tenants.findAll().stream().map(tenant -> Map.of("id", tenant.getId(), "name", tenant.getDisplayName(), "created_at", tenant.getCreatedAt())).toList();
    }

    @PostMapping("/tenants")
    public Object createTenant(@AuthenticationPrincipal AppUser admin, @Valid @RequestBody CreateTenantRequest request) {
        requirePlatformAdmin(admin);
        if (!request.tenantId().matches("[A-Za-z0-9_-]{2,64}")) throw new IllegalArgumentException("租户标识只能包含字母、数字、下划线和短横线");
        if (tenants.existsById(request.tenantId())) throw new ResponseStatusException(HttpStatus.CONFLICT, "租户标识已存在");
        if (users.existsByUsername(request.adminUsername())) throw new ResponseStatusException(HttpStatus.CONFLICT, "管理员账号已存在");
        Tenant tenant = new Tenant(); tenant.setId(request.tenantId()); tenant.setDisplayName(request.displayName().trim()); tenants.save(tenant);
        AppUser tenantAdmin = new AppUser(); tenantAdmin.setUsername(request.adminUsername().trim());
        tenantAdmin.setDisplayName(request.adminDisplayName().trim()); tenantAdmin.setPasswordHash(passwords.encode(request.adminPassword()));
        tenantAdmin.setRole(UserRole.ADMIN); tenantAdmin.setTenantId(request.tenantId());
        AppUser saved = users.save(tenantAdmin);
        return Map.of("tenant_id", tenant.getId(), "tenant_name", tenant.getDisplayName(), "admin_user_id", saved.getId());
    }

    private void requirePlatformAdmin(AppUser user) {
        if (!user.isPlatformAdmin()) throw new ResponseStatusException(HttpStatus.FORBIDDEN, "仅平台管理员可以管理租户");
    }

    public record CreateUserRequest(@NotBlank @Size(max = 80) String username, @NotBlank @Size(min = 12, max = 128) String password,
                                    @NotBlank @Size(max = 120) String displayName, UserRole role) {}
    public record CreateTenantRequest(@NotBlank @Size(min = 2, max = 64) String tenantId,
                                      @NotBlank @Size(max = 120) String displayName,
                                      @NotBlank @Size(max = 80) String adminUsername,
                                      @NotBlank @Size(min = 12, max = 128) String adminPassword,
                                      @NotBlank @Size(max = 120) String adminDisplayName) {}
}
