package com.hypersale.raglogin.controller;

import com.hypersale.raglogin.config.RagProperties;
import com.hypersale.raglogin.model.AppUser;
import com.hypersale.raglogin.model.UserRole;
import com.hypersale.raglogin.model.Tenant;
import com.hypersale.raglogin.repository.AppUserRepository;
import com.hypersale.raglogin.repository.TenantRepository;
import com.hypersale.raglogin.security.JwtService;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.util.Map;

@RestController
@RequestMapping("/api/auth")
public class AuthController {
    private static final Object BOOTSTRAP_LOCK = new Object();
    private final AppUserRepository users;
    private final PasswordEncoder passwords;
    private final JwtService tokens;
    private final RagProperties properties;
    private final TenantRepository tenants;

    public AuthController(AppUserRepository users, PasswordEncoder passwords, JwtService tokens, RagProperties properties, TenantRepository tenants) {
        this.users = users;
        this.passwords = passwords;
        this.tokens = tokens;
        this.properties = properties;
        this.tenants = tenants;
    }

    @PostMapping("/bootstrap")
    public ResponseEntity<?> bootstrap(@Valid @RequestBody BootstrapRequest request) {
        synchronized (BOOTSTRAP_LOCK) {
            if (users.count() != 0) throw new ResponseStatusException(HttpStatus.CONFLICT, "初始化账号已创建");
            AppUser admin = new AppUser();
            admin.setUsername(request.username().trim());
            admin.setDisplayName(request.displayName().trim());
            admin.setPasswordHash(passwords.encode(request.password()));
            admin.setRole(UserRole.ADMIN);
            admin.setTenantId(properties.getDefaultTenant());
            admin.setPlatformAdmin(true);
            if (!tenants.existsById(properties.getDefaultTenant())) {
                Tenant tenant = new Tenant(); tenant.setId(properties.getDefaultTenant()); tenant.setDisplayName("默认企业"); tenants.save(tenant);
            }
            return ResponseEntity.status(HttpStatus.CREATED).body(loginPayload(users.save(admin)));
        }
    }

    @PostMapping("/login")
    public Map<String, Object> login(@Valid @RequestBody LoginRequest request) {
        AppUser user = users.findByUsername(request.username().trim())
                .filter(AppUser::isEnabled)
                .filter(candidate -> passwords.matches(request.password(), candidate.getPasswordHash()))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "用户名或密码错误"));
        return loginPayload(user);
    }

    @GetMapping("/me")
    public Map<String, Object> me(@AuthenticationPrincipal AppUser user) { return userView(user); }

    @PostMapping("/password")
    public Map<String, String> changePassword(@AuthenticationPrincipal AppUser user, @Valid @RequestBody ChangePasswordRequest request) {
        if (!passwords.matches(request.currentPassword(), user.getPasswordHash())) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "当前密码不正确");
        }
        user.setPasswordHash(passwords.encode(request.newPassword()));
        users.save(user);
        return Map.of("status", "updated");
    }

    private Map<String, Object> loginPayload(AppUser user) {
        return Map.of("token", tokens.issue(user), "user", userView(user));
    }

    private Map<String, Object> userView(AppUser user) {
        return Map.of("id", user.getId(), "username", user.getUsername(), "display_name", user.getDisplayName(),
                "role", user.getRole().name(), "tenant_id", user.getTenantId(), "platform_admin", user.isPlatformAdmin());
    }

    public record BootstrapRequest(@NotBlank @Size(max = 80) String username,
                                   @NotBlank @Size(min = 12, max = 128) String password,
                                   @NotBlank @Size(max = 120) String displayName) {}
    public record LoginRequest(@NotBlank String username, @NotBlank String password) {}
    public record ChangePasswordRequest(@NotBlank String currentPassword, @NotBlank @Size(min = 12, max = 128) String newPassword) {}
}
