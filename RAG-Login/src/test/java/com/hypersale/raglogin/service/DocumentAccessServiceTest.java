package com.hypersale.raglogin.service;

import com.hypersale.raglogin.model.AppUser;
import com.hypersale.raglogin.model.DocumentVisibility;
import com.hypersale.raglogin.model.KnowledgeDocument;
import com.hypersale.raglogin.model.UserRole;
import com.hypersale.raglogin.repository.KnowledgeDocumentRepository;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class DocumentAccessServiceTest {
    private final DocumentAccessService service = new DocumentAccessService(mock(KnowledgeDocumentRepository.class));

    @Test
    void publicDocumentsAreLimitedToTheirTenant() {
        AppUser user = user(12L, "tenant-a", UserRole.USER);
        KnowledgeDocument document = document("tenant-a", 99L, DocumentVisibility.TENANT_PUBLIC);
        assertTrue(service.canRead(user, document));
        document.setTenantId("tenant-b");
        assertFalse(service.canRead(user, document));
    }

    @Test
    void selectedDocumentsRequireAnExplicitUserGrant() {
        AppUser user = user(12L, "tenant-a", UserRole.USER);
        KnowledgeDocument document = document("tenant-a", 99L, DocumentVisibility.SELECTED_USERS);
        assertFalse(service.canRead(user, document));
        document.setGrantedUserIds(Set.of(12L));
        assertTrue(service.canRead(user, document));
    }

    @Test
    void privateDocumentsBelongToTheirUploaderAndTenantAdmins() {
        KnowledgeDocument document = document("tenant-a", 12L, DocumentVisibility.PRIVATE);
        assertTrue(service.canRead(user(12L, "tenant-a", UserRole.USER), document));
        assertFalse(service.canRead(user(13L, "tenant-a", UserRole.USER), document));
        assertTrue(service.canRead(user(13L, "tenant-a", UserRole.ADMIN), document));
        assertFalse(service.canRead(user(13L, "tenant-b", UserRole.ADMIN), document));
    }

    private static AppUser user(long id, String tenantId, UserRole role) {
        AppUser user = new AppUser();
        try {
            var field = AppUser.class.getDeclaredField("id");
            field.setAccessible(true);
            field.set(user, id);
        } catch (ReflectiveOperationException exception) {
            throw new AssertionError(exception);
        }
        user.setTenantId(tenantId);
        user.setRole(role);
        return user;
    }

    private static KnowledgeDocument document(String tenantId, Long ownerId, DocumentVisibility visibility) {
        KnowledgeDocument document = new KnowledgeDocument();
        document.setTenantId(tenantId);
        document.setOwnerUserId(ownerId);
        document.setVisibility(visibility);
        return document;
    }
}
