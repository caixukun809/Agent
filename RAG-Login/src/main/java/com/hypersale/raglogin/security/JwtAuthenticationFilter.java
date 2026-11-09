package com.hypersale.raglogin.security;

import com.hypersale.raglogin.model.AppUser;
import com.hypersale.raglogin.repository.AppUserRepository;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Map;

@Component
public class JwtAuthenticationFilter extends OncePerRequestFilter {
    private final JwtService jwtService;
    private final AppUserRepository users;

    public JwtAuthenticationFilter(JwtService jwtService, AppUserRepository users) {
        this.jwtService = jwtService;
        this.users = users;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        String authorization = request.getHeader("Authorization");
        if (authorization != null && authorization.startsWith("Bearer ")
                && SecurityContextHolder.getContext().getAuthentication() == null) {
            Map<String, Object> claims = jwtService.verify(authorization.substring(7));
            if (claims != null) {
                String username = String.valueOf(claims.getOrDefault("sub", ""));
                AppUser user = users.findByUsername(username).orElse(null);
                if (user != null && user.isEnabled()
                        && user.getRole().name().equals(claims.get("role"))
                        && user.getTenantId().equals(claims.get("tenant"))) {
                    var authentication = new UsernamePasswordAuthenticationToken(user, null, user.getAuthorities());
                    SecurityContextHolder.getContext().setAuthentication(authentication);
                }
            }
        }
        chain.doFilter(request, response);
    }
}
