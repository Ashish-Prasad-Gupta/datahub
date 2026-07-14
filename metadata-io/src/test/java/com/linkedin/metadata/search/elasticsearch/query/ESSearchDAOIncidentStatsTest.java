package com.linkedin.metadata.search.elasticsearch.query;

import static io.datahubproject.test.search.SearchTestUtils.TEST_OS_SEARCH_CONFIG;
import static io.datahubproject.test.search.SearchTestUtils.TEST_SEARCH_SERVICE_CONFIG;
import static org.testng.Assert.assertTrue;

import com.linkedin.common.urn.Urn;
import com.linkedin.metadata.search.elasticsearch.query.filter.QueryFilterRewriteChain;
import com.linkedin.metadata.utils.elasticsearch.SearchClientShim;
import io.datahubproject.metadata.context.OperationContext;
import io.datahubproject.test.metadata.context.TestOperationContexts;
import java.util.Set;
import org.mockito.Mockito;
import org.opensearch.action.search.SearchRequest;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

public class ESSearchDAOIncidentStatsTest {

  private ESSearchDAO esSearchDAO;
  private OperationContext opContext;

  @BeforeMethod
  public void setUp() {
    SearchClientShim<?> mockClient = Mockito.mock(SearchClientShim.class);
    opContext = TestOperationContexts.systemContextNoSearchAuthorization();
    esSearchDAO =
        new ESSearchDAO(
            mockClient,
            false,
            TEST_OS_SEARCH_CONFIG,
            null,
            QueryFilterRewriteChain.EMPTY,
            TEST_SEARCH_SERVICE_CONFIG);
  }

  @Test
  public void testBuildActiveIncidentStatsRequestShape() throws Exception {
    final Set<Urn> urns =
        Set.of(
            Urn.createFromString("urn:li:dataset:(urn:li:dataPlatform:x,a,PROD)"),
            Urn.createFromString("urn:li:dataset:(urn:li:dataPlatform:x,b,PROD)"));

    final SearchRequest request = esSearchDAO.buildActiveIncidentStatsRequest(opContext, urns);
    final String source = request.source().toString();

    assertTrue(source.contains("\"terms\""), "expected terms aggregation");
    assertTrue(source.contains("entities.keyword"), "expected group-by on entities.keyword");
    assertTrue(source.contains("\"top_hits\""), "expected top_hits sub-agg for latest incident");
    assertTrue(source.contains("lastUpdated"), "expected sort by lastUpdated");
    assertTrue(source.contains("ACTIVE"), "expected active-state filter");
  }
}
